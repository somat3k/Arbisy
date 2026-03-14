// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title IFlashLoanReceiver
 * @notice Aave V3 flash loan receiver interface.
 *         The contract receiving a flash loan must implement executeOperation.
 */
interface IFlashLoanReceiver {
    /**
     * @notice Called by the Aave Pool after transferring the requested assets.
     * @param assets      Addresses of the flash-loaned assets.
     * @param amounts     Amounts of each asset that was flash-loaned.
     * @param premiums    Fee owed for each asset (amounts[i] * 0.05%).
     * @param initiator   Address that triggered the flash loan.
     * @param params      Arbitrary encoded parameters passed by the caller.
     * @return True if the operation succeeded; the Pool will pull back
     *         amounts[i] + premiums[i] from this contract.
     */
    function executeOperation(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        address initiator,
        bytes calldata params
    ) external returns (bool);
}
