// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title ArbitrageLib
 * @notice Pure helper functions for on-chain arbitrage calculations.
 */
library ArbitrageLib {
    uint256 internal constant BPS_DENOMINATOR = 10_000;
    uint256 internal constant FLASH_LOAN_PREMIUM_BPS = 5; // 0.05%

    /**
     * @notice Compute the minimum amount that must be repaid to Aave for a
     *         flash loan of `amount` at the standard 0.05% premium.
     */
    function repayAmount(uint256 amount) internal pure returns (uint256) {
        return amount + flashLoanFee(amount);
    }

    /**
     * @notice Compute the flat fee for a flash loan of `amount`.
     */
    function flashLoanFee(uint256 amount) internal pure returns (uint256) {
        return (amount * FLASH_LOAN_PREMIUM_BPS) / BPS_DENOMINATOR;
    }

    /**
     * @notice Compute net profit = outputAmount - repayAmount.
     *         Returns 0 if the trade is unprofitable.
     */
    function netProfit(
        uint256 loanAmount,
        uint256 outputAmount
    ) internal pure returns (uint256 profit, bool profitable) {
        uint256 repay = repayAmount(loanAmount);
        if (outputAmount > repay) {
            profit = outputAmount - repay;
            profitable = true;
        } else {
            profit = 0;
            profitable = false;
        }
    }

    /**
     * @notice Apply a DEX fee (in BPS) to an input amount and return the
     *         effective output amount assuming a fixed fee model.
     */
    function applyFee(
        uint256 amountIn,
        uint256 feeBps
    ) internal pure returns (uint256 amountAfterFee) {
        amountAfterFee = (amountIn * (BPS_DENOMINATOR - feeBps)) / BPS_DENOMINATOR;
    }

    /**
     * @notice Decode an arbitrage swap path encoded as:
     *         abi.encode(address[] dexRouters, address[] tokens, uint24[] fees, bool[] isV3)
     */
    function decodeSwapPath(bytes calldata params)
        internal
        pure
        returns (
            address[] memory dexRouters,
            address[] memory tokens,
            uint24[]  memory fees,
            bool[]    memory isV3
        )
    {
        (dexRouters, tokens, fees, isV3) = abi.decode(
            params,
            (address[], address[], uint24[], bool[])
        );
    }

    /**
     * @notice Decode a swap path that includes a replay-protection nonce and
     *         per-hop expected output amounts for slippage protection (E7-S2, E7-S5).
     *
     * Encoding:
     *   abi.encode(bytes32 nonce, address[] dexRouters, address[] tokens,
     *              uint24[] fees, bool[] isV3, uint256[] expectedAmountsOut)
     *
     * expectedAmountsOut[hop] is the off-chain simulated output for that hop
     * (in tokenOut units).  A zero entry means no minimum is enforced for that hop.
     */
    function decodeSwapPathWithNonce(bytes calldata params)
        internal
        pure
        returns (
            bytes32   nonce,
            address[] memory dexRouters,
            address[] memory tokens,
            uint24[]  memory fees,
            bool[]    memory isV3,
            uint256[] memory expectedAmountsOut
        )
    {
        (nonce, dexRouters, tokens, fees, isV3, expectedAmountsOut) = abi.decode(
            params,
            (bytes32, address[], address[], uint24[], bool[], uint256[])
        );
    }
}
