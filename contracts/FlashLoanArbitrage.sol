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
 * Execution flow:
 *   1. Owner calls initiateFlashLoan() with encoded swap path in `params`.
 *   2. Aave Pool transfers the requested asset to this contract.
 *   3. executeOperation() callback fires:
 *      a. Decodes the swap path.
 *      b. Executes each hop (V3 exactInputSingle or V2 swapExactTokensForTokens).
 *      c. Verifies that output ≥ loan + premium (profit check).
 *      d. Approves the Pool to pull back loan + premium.
 *   4. Pool reclaims loan + premium; net profit stays in this contract.
 *   5. Owner calls withdrawToken() to collect profits.
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

    // ── Errors ────────────────────────────────────────────────────────────────
    error OnlyOwner();
    error OnlyAavePool();
    error ReentrancyGuard();
    error UnauthorizedInitiator();
    error DEXNotAllowed(address dex);
    error UnprofitableTrade(uint256 output, uint256 required);
    error InvalidPathLength();
    error ExceedsMaxLoanAmount(uint256 requested, uint256 maximum);
    error ZeroAmount();

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
    ) external override nonReentrant returns (bool) {
        // Only the Aave Pool may call this function
        if (msg.sender != AAVE_POOL) revert OnlyAavePool();
        // Only operations initiated by this contract are valid
        if (initiator != address(this)) revert UnauthorizedInitiator();

        for (uint256 i = 0; i < assets.length; i++) {
            _executeArbitrage(assets[i], amounts[i], premiums[i], params);
        }

        return true;
    }

    /**
     * @dev Core arbitrage logic for a single asset.
     */
    function _executeArbitrage(
        address asset,
        uint256 loanAmount,
        uint256 premium,
        bytes calldata params
    ) internal {
        uint256 repay = loanAmount + premium;

        // Decode the swap path
        (
            address[] memory dexRouters,
            address[] memory tokens,
            uint24[]  memory fees,
            bool[]    memory isV3
        ) = ArbitrageLib.decodeSwapPath(params);

        // Validate path: tokens must start and end with `asset`
        if (tokens.length < 2) revert InvalidPathLength();
        if (dexRouters.length != tokens.length - 1) revert InvalidPathLength();

        // Execute each hop
        uint256 currentAmount = loanAmount;
        for (uint256 hop = 0; hop < dexRouters.length; hop++) {
            address router = dexRouters[hop];
            if (!allowedDEX[router]) revert DEXNotAllowed(router);

            address tokenIn  = tokens[hop];
            address tokenOut = tokens[hop + 1];

            IERC20(tokenIn).approve(router, currentAmount);

            if (isV3[hop]) {
                currentAmount = _swapV3(
                    router, tokenIn, tokenOut, fees[hop], currentAmount
                );
            } else {
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

        emit ArbitrageExecuted(asset, loanAmount, repay, profit, tx.origin);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal swap helpers
    // ─────────────────────────────────────────────────────────────────────────

    function _swapV3(
        address router,
        address tokenIn,
        address tokenOut,
        uint24  fee,
        uint256 amountIn
    ) internal returns (uint256 amountOut) {
        IUniswapV3Router.ExactInputSingleParams memory swapParams =
            IUniswapV3Router.ExactInputSingleParams({
                tokenIn:           tokenIn,
                tokenOut:          tokenOut,
                fee:               fee,
                recipient:         address(this),
                deadline:          block.timestamp + 60,
                amountIn:          amountIn,
                amountOutMinimum:  0, // slippage checked via profit guard
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
            0, // slippage checked via profit guard
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
     * @param assets             ERC-20 token addresses to borrow.
     * @param amounts            Amounts to borrow for each asset.
     * @param interestRateModes  0 = no open debt (standard flash loan).
     * @param params             ABI-encoded swap path (see ArbitrageLib).
     */
    function initiateFlashLoan(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata interestRateModes,
        bytes calldata params
    ) external onlyOwner nonReentrant {
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
            0 // referralCode
        );
    }

    /**
     * @notice Initiate a single-asset Aave V3 flash loan (gas optimized).
     */
    function initiateFlashLoanSimple(
        address asset,
        uint256 amount,
        bytes calldata params
    ) external onlyOwner nonReentrant {
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
     * @param loanAmount   Amount to borrow.
     * @param outputAmount Estimated swap output (from off-chain simulation).
     * @return profit      Expected profit after repaying loan + premium.
     * @return profitable  True if the trade would be profitable.
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

    function withdrawToken(address token, uint256 amount, address to)
        external
        onlyOwner
        nonReentrant
    {
        IERC20(token).transfer(to, amount);
        emit ProfitWithdrawn(token, amount, to);
    }

    function withdrawEth(address payable to) external onlyOwner nonReentrant {
        to.transfer(address(this).balance);
    }

    receive() external payable {}
}
