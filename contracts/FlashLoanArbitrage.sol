// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./interfaces/IFlashLoanReceiver.sol";
import "./interfaces/IDEXRouter.sol";
import "./libraries/ArbitrageLib.sol";

// ── Minimal ERC-20 interface ──────────────────────────────────────────────────
interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

// ── Minimal Aave V3 Pool interface ────────────────────────────────────────────
interface IPool {
    function flashLoan(
        address receiverAddress,
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata interestRateModes,
        address onBehalfOf,
        bytes calldata params,
        uint16 referralCode
    ) external;

    function flashLoanSimple(
        address receiverAddress,
        address asset,
        uint256 amount,
        bytes calldata params,
        uint16 referralCode
    ) external;
}

/**
 * @title FlashLoanArbitrage
 * @notice Executes atomic arbitrage across multiple DEXes funded by an
 *         Aave V3 flash loan on Polygon zkEVM.
 *
 * Security hardening:
 *   E7-S2  Replay protection — nonce embedded in params; used nonces are
 *          permanently recorded and rejected on re-use.
 *   E7-S3  Pause mechanism  — owner can halt all execution instantly.
 *   E7-S4  withdrawToken allow-list — profits can only be swept to
 *          pre-approved destination addresses.
 *   E7-S5  maxSlippageBps   — per-swap amountOutMinimum enforced in V3 hops.
 *
 * Execution flow:
 *   1. Owner calls initiateFlashLoanSimple() with encoded swap path in `params`.
 *   2. Aave Pool transfers the requested asset to this contract.
 *   3. executeOperation() callback fires:
 *      a. Checks contract is not paused (E7-S3).
 *      b. Validates and consumes the nonce (E7-S2).
 *      c. Decodes the swap path.
 *      d. Executes each hop; for V3 hops amountOutMinimum is applied (E7-S5).
 *      e. Verifies that output ≥ loan + premium (profit check).
 *      f. Approves the Pool to pull back loan + premium.
 *   4. Pool reclaims loan + premium; net profit stays in this contract.
 *   5. Owner calls withdrawToken() to an allow-listed address (E7-S4).
 */
contract FlashLoanArbitrage is IFlashLoanReceiver {
    using ArbitrageLib for uint256;

    // ── Immutable state ───────────────────────────────────────────────────────
    address public immutable AAVE_POOL;
    address public immutable owner;

    // ── Reentrancy guard ──────────────────────────────────────────────────────
    uint256 private _status;
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED     = 2;

    // ── DEX whitelist ─────────────────────────────────────────────────────────
    mapping(address => bool) public allowedDEX;

    // ── E7-S2: Replay-protection nonce registry ───────────────────────────────
    mapping(bytes32 => bool) public usedNonces;

    // ── E7-S3: Pause mechanism ────────────────────────────────────────────────
    bool public paused;

    // ── E7-S4: Withdrawal allow-list ──────────────────────────────────────────
    mapping(address => bool) public allowedWithdrawTo;

    // ── E7-S5: Max per-hop slippage ───────────────────────────────────────────
    /// @notice Maximum allowed slippage in basis points per swap hop (default 50 = 0.5%).
    uint256 public maxSlippageBps = 50;

    // ── Config ────────────────────────────────────────────────────────────────
    uint256 public maxFlashLoanAmount = 1_000_000e18; // 1M tokens default cap

    // ── Events ────────────────────────────────────────────────────────────────
    event ArbitrageExecuted(
        address indexed asset,
        uint256 loanAmount,
        uint256 repayAmount,
        uint256 netProfit,
        address indexed initiator
    );
    event DEXAllowlistUpdated(address indexed dex, bool allowed);
    event MaxLoanAmountUpdated(uint256 newMax);
    event ProfitWithdrawn(address indexed token, uint256 amount, address indexed to);
    event Paused(address by);
    event Unpaused(address by);
    event WithdrawAllowlistUpdated(address indexed addr, bool allowed);
    event MaxSlippageUpdated(uint256 newMaxSlippageBps);

    // ── Errors ────────────────────────────────────────────────────────────────
    error OnlyOwner();
    error OnlyAavePool();
    error ReentrancyGuard();
    error UnauthorizedInitiator();
    error DEXNotAllowed(address dex);
    error UnprofitableTrade(uint256 output, uint256 required);
    error InvalidPathLength();
    error InvalidPathAsset(address expected, address got);
    error ArrayLengthMismatch();
    error MultiAssetNotSupported();
    error ExceedsMaxLoanAmount(uint256 requested, uint256 maximum);
    error ZeroAmount();
    error ContractPaused();                              // E7-S3
    error NonceAlreadyUsed(bytes32 nonce);              // E7-S2
    error WithdrawDestinationNotAllowed(address to);    // E7-S4
    error SlippageExceeded(uint256 amountOut, uint256 minimum); // E7-S5
    error SlippageCapTooHigh(uint256 provided, uint256 maximum); // E7-S5

    // ── Modifiers ─────────────────────────────────────────────────────────────
    modifier onlyOwner() {
        if (msg.sender != owner) revert OnlyOwner();
        _;
    }

    modifier nonReentrant() {
        if (_status == _ENTERED) revert ReentrancyGuard();
        _status = _ENTERED;
        _;
        _status = _NOT_ENTERED;
    }

    /// @dev E7-S3: Reverts if the contract is paused.
    modifier whenNotPaused() {
        if (paused) revert ContractPaused();
        _;
    }

    // ── Constructor ───────────────────────────────────────────────────────────
    constructor(address aavePool) {
        AAVE_POOL = aavePool;
        owner     = msg.sender;
        _status   = _NOT_ENTERED;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // IFlashLoanReceiver — Aave V3 callback
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * @inheritdoc IFlashLoanReceiver
     * @dev Called by the Aave Pool after transferring flash-loaned assets.
     *      Performs swaps and repays the loan + premium atomically.
     */
    function executeOperation(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        address initiator,
        bytes calldata params
    ) external override nonReentrant whenNotPaused returns (bool) {
        // Only the Aave Pool may call this function
        if (msg.sender != AAVE_POOL) revert OnlyAavePool();
        // Only operations initiated by this contract are valid
        if (initiator != address(this)) revert UnauthorizedInitiator();
        // This contract supports single-asset flash loans only.
        // Multi-asset borrows would require per-asset path encoding in params.
        if (assets.length != 1) revert MultiAssetNotSupported();

        _executeArbitrage(assets[0], amounts[0], premiums[0], params);

        return true;
    }

    /**
     * @dev Core arbitrage logic for a single asset.
     *      Includes nonce replay check (E7-S2) and per-hop slippage guard (E7-S5).
     */
    function _executeArbitrage(
        address asset,
        uint256 loanAmount,
        uint256 premium,
        bytes calldata params
    ) internal {
        uint256 repay = loanAmount + premium;

        // E7-S2: Decode and consume the nonce embedded in params.
        // Params layout: abi.encode(bytes32 nonce, address[] routers,
        //                           address[] tokens, uint24[] fees, bool[] isV3,
        //                           uint256[] expectedAmountsOut)
        // expectedAmountsOut[hop] is the off-chain simulated output for that hop.
        // A zero entry means "no minimum enforced for this hop" (profit guard is backstop).
        (
            bytes32 nonce,
            address[] memory dexRouters,
            address[] memory tokens,
            uint24[]  memory fees,
            bool[]    memory isV3,
            uint256[] memory expectedAmountsOut
        ) = ArbitrageLib.decodeSwapPathWithNonce(params);

        if (usedNonces[nonce]) revert NonceAlreadyUsed(nonce);
        usedNonces[nonce] = true;

        // Validate path lengths and asset in/out invariant.
        if (tokens.length < 2) revert InvalidPathLength();
        if (dexRouters.length != tokens.length - 1) revert InvalidPathLength();
        if (fees.length != dexRouters.length) revert ArrayLengthMismatch();
        if (isV3.length != dexRouters.length) revert ArrayLengthMismatch();
        if (tokens[0] != asset) revert InvalidPathAsset(asset, tokens[0]);
        if (tokens[tokens.length - 1] != asset)
            revert InvalidPathAsset(asset, tokens[tokens.length - 1]);

        // Execute each hop
        uint256 currentAmount = loanAmount;
        for (uint256 hop = 0; hop < dexRouters.length; hop++) {
            address router = dexRouters[hop];
            if (!allowedDEX[router]) revert DEXNotAllowed(router);

            address tokenIn  = tokens[hop];
            address tokenOut = tokens[hop + 1];

            IERC20(tokenIn).approve(router, currentAmount);

            if (isV3[hop]) {
                // E7-S5: Use off-chain expected output as amountOutMinimum,
                // reduced by maxSlippageBps. This correctly computes the minimum
                // in tokenOut units using the simulated exchange rate.
                // If no expected output was provided (zero), fall back to 0
                // (the overall profit guard at the end is the backstop).
                uint256 amountOutMin = 0;
                if (expectedAmountsOut.length > hop && expectedAmountsOut[hop] > 0) {
                    amountOutMin = (expectedAmountsOut[hop] * (10_000 - maxSlippageBps)) / 10_000;
                }
                uint256 amountOut = _swapV3(
                    router, tokenIn, tokenOut, fees[hop], currentAmount, amountOutMin
                );
                if (amountOutMin > 0 && amountOut < amountOutMin)
                    revert SlippageExceeded(amountOut, amountOutMin);
                currentAmount = amountOut;
            } else {
                // V2 uses amountOutMin = 0 (profit guard is the backstop)
                currentAmount = _swapV2(
                    router, tokenIn, tokenOut, currentAmount
                );
            }
        }

        // Profit check — revert entire transaction if unprofitable
        if (currentAmount < repay) {
            revert UnprofitableTrade(currentAmount, repay);
        }

        uint256 profit = currentAmount - repay;

        // Approve the Aave Pool to pull back loan + premium
        IERC20(asset).approve(AAVE_POOL, repay);

        emit ArbitrageExecuted(asset, loanAmount, repay, profit, owner);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal swap helpers
    // ─────────────────────────────────────────────────────────────────────────

    function _swapV3(
        address router,
        address tokenIn,
        address tokenOut,
        uint24  fee,
        uint256 amountIn,
        uint256 amountOutMinimum  // E7-S5
    ) internal returns (uint256 amountOut) {
        IUniswapV3Router.ExactInputSingleParams memory swapParams =
            IUniswapV3Router.ExactInputSingleParams({
                tokenIn:           tokenIn,
                tokenOut:          tokenOut,
                fee:               fee,
                recipient:         address(this),
                deadline:          block.timestamp + 60,
                amountIn:          amountIn,
                amountOutMinimum:  amountOutMinimum,
                sqrtPriceLimitX96: 0
            });
        amountOut = IUniswapV3Router(router).exactInputSingle(swapParams);
    }

    function _swapV2(
        address router,
        address tokenIn,
        address tokenOut,
        uint256 amountIn
    ) internal returns (uint256 amountOut) {
        address[] memory path = new address[](2);
        path[0] = tokenIn;
        path[1] = tokenOut;

        uint256[] memory amounts = IUniswapV2Router(router).swapExactTokensForTokens(
            amountIn,
            0, // slippage checked via overall profit guard
            path,
            address(this),
            block.timestamp + 60
        );
        amountOut = amounts[amounts.length - 1];
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public entry points
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * @notice Initiate a multi-asset Aave V3 flash loan.
     */
    function initiateFlashLoan(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata interestRateModes,
        bytes calldata params
    ) external onlyOwner nonReentrant whenNotPaused {
        for (uint256 i = 0; i < amounts.length; i++) {
            if (amounts[i] == 0) revert ZeroAmount();
            if (amounts[i] > maxFlashLoanAmount)
                revert ExceedsMaxLoanAmount(amounts[i], maxFlashLoanAmount);
        }

        IPool(AAVE_POOL).flashLoan(
            address(this),
            assets,
            amounts,
            interestRateModes,
            address(this),
            params,
            0
        );
    }

    /**
     * @notice Initiate a single-asset Aave V3 flash loan (gas optimized).
     */
    function initiateFlashLoanSimple(
        address asset,
        uint256 amount,
        bytes calldata params
    ) external onlyOwner nonReentrant whenNotPaused {
        if (amount == 0) revert ZeroAmount();
        if (amount > maxFlashLoanAmount)
            revert ExceedsMaxLoanAmount(amount, maxFlashLoanAmount);

        IPool(AAVE_POOL).flashLoanSimple(
            address(this),
            asset,
            amount,
            params,
            0
        );
    }

    /**
     * @notice Pure view simulation of expected net profit.
     */
    function simulateProfit(
        uint256 loanAmount,
        uint256 outputAmount
    ) external pure returns (uint256 profit, bool profitable) {
        (profit, profitable) = ArbitrageLib.netProfit(loanAmount, outputAmount);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Admin functions
    // ─────────────────────────────────────────────────────────────────────────

    function setAllowedDEX(address dex, bool allowed) external onlyOwner {
        allowedDEX[dex] = allowed;
        emit DEXAllowlistUpdated(dex, allowed);
    }

    function setMaxFlashLoanAmount(uint256 newMax) external onlyOwner {
        maxFlashLoanAmount = newMax;
        emit MaxLoanAmountUpdated(newMax);
    }

    /// @notice E7-S4: Withdraw profits to an allow-listed address only.
    function withdrawToken(address token, uint256 amount, address to)
        external
        onlyOwner
        nonReentrant
    {
        if (!allowedWithdrawTo[to]) revert WithdrawDestinationNotAllowed(to);
        IERC20(token).transfer(to, amount);
        emit ProfitWithdrawn(token, amount, to);
    }

    function withdrawEth(address payable to) external onlyOwner nonReentrant {
        if (!allowedWithdrawTo[to]) revert WithdrawDestinationNotAllowed(to);
        to.transfer(address(this).balance);
    }

    /// @notice E7-S4: Add or remove an address from the withdrawal allow-list.
    function setAllowedWithdrawTo(address addr, bool allowed) external onlyOwner {
        allowedWithdrawTo[addr] = allowed;
        emit WithdrawAllowlistUpdated(addr, allowed);
    }

    /// @notice E7-S3: Pause all flash-loan execution.
    function pause() external onlyOwner {
        paused = true;
        emit Paused(msg.sender);
    }

    /// @notice E7-S3: Resume flash-loan execution.
    function unpause() external onlyOwner {
        paused = false;
        emit Unpaused(msg.sender);
    }

    /// @notice E7-S5: Update the per-hop maximum slippage (in basis points).
    function setMaxSlippageBps(uint256 newMaxSlippageBps) external onlyOwner {
        if (newMaxSlippageBps > 1_000) revert SlippageCapTooHigh(newMaxSlippageBps, 1_000); // max 10%
        maxSlippageBps = newMaxSlippageBps;
        emit MaxSlippageUpdated(newMaxSlippageBps);
    }

    receive() external payable {}
}

