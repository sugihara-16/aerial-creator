from __future__ import annotations

"""Explicit C3 action contracts used by matched pi_L ablations."""


ORDER9_C3_ACTION_CONTRACT_COMPRESSION_ONLY = "contact_compression_only"
ORDER9_C3_ACTION_CONTRACT_COMPRESSION_PLUS_CENTROIDAL = (
    "contact_compression_plus_centroidal"
)
ORDER9_C3_ACTION_CONTRACT_COMPRESSION_PLUS_GLOBAL = (
    "contact_compression_plus_global"
)
ORDER9_C3_ACTION_CONTRACT_FULL_INDEPENDENT_COMPRESSION = (
    "full_policy_command_plus_independent_compression"
)
ORDER9_C3_ACTION_CONTRACT_CONTACT_SPACE_PROJECTED = (
    "contact_space_projected_policy_command"
)
ORDER9_C3_ACTION_CONTRACTS = (
    ORDER9_C3_ACTION_CONTRACT_COMPRESSION_ONLY,
    ORDER9_C3_ACTION_CONTRACT_COMPRESSION_PLUS_CENTROIDAL,
    ORDER9_C3_ACTION_CONTRACT_COMPRESSION_PLUS_GLOBAL,
    ORDER9_C3_ACTION_CONTRACT_FULL_INDEPENDENT_COMPRESSION,
    ORDER9_C3_ACTION_CONTRACT_CONTACT_SPACE_PROJECTED,
)


def order9_c3_action_contract_global_dimension(contract: str) -> int:
    """Return the enabled PolicyCommand-global prefix for one contract."""

    if contract == ORDER9_C3_ACTION_CONTRACT_COMPRESSION_ONLY:
        return 0
    if contract == ORDER9_C3_ACTION_CONTRACT_COMPRESSION_PLUS_CENTROIDAL:
        # Body-pose and body-twist correction; residual wrench is excluded.
        return 12
    if contract == ORDER9_C3_ACTION_CONTRACT_COMPRESSION_PLUS_GLOBAL:
        # Body-pose, body-twist, and residual-wrench correction.
        return 18
    if contract == ORDER9_C3_ACTION_CONTRACT_FULL_INDEPENDENT_COMPRESSION:
        # Complete PolicyCommand: body pose/twist, residual wrench, every
        # local-joint residual, and the separate coordinated-compression gain.
        return 18
    if contract == ORDER9_C3_ACTION_CONTRACT_CONTACT_SPACE_PROJECTED:
        # Centroidal pose/twist are projected to the active-contact-compatible
        # subspace.  Residual wrench is not a learned v7 coordinate.
        return 12
    raise ValueError(f"invalid Order 9 C3 action contract: {contract!r}")


def order9_c3_action_contract_uses_full_policy(contract: str) -> bool:
    """Return whether no global or per-joint actor coordinate is masked."""

    if contract not in ORDER9_C3_ACTION_CONTRACTS:
        raise ValueError(f"invalid Order 9 C3 action contract: {contract!r}")
    return contract in {
        ORDER9_C3_ACTION_CONTRACT_FULL_INDEPENDENT_COMPRESSION,
        ORDER9_C3_ACTION_CONTRACT_CONTACT_SPACE_PROJECTED,
    }


def order9_c3_action_contract_uses_contact_space(contract: str) -> bool:
    if contract not in ORDER9_C3_ACTION_CONTRACTS:
        raise ValueError(f"invalid Order 9 C3 action contract: {contract!r}")
    return contract == ORDER9_C3_ACTION_CONTRACT_CONTACT_SPACE_PROJECTED
