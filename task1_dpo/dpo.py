from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    """Return scalar DPO loss plus lightweight diagnostics.

    Validate this implementation against the equation in the assignment manual
    before using it.
    """

    policy_margin = (
        policy_chosen_logp
        - policy_rejected_logp
    )

    ref_margin = (
        ref_chosen_logp
        - ref_rejected_logp
    )

    # FIX (validated against the DPO objective): the implicit-reward logit is
    #   beta * ([log pi(y+) - log ref(y+)] - [log pi(y-) - log ref(y-)])
    #   = beta * (policy_margin - ref_margin).
    # The starter code ADDED the reference margin, which rewards the policy for
    # agreeing with the reference instead of measuring movement away from it.
    logits = beta * (
        policy_margin - ref_margin
    )

    loss = -F.logsigmoid(
        logits
    ).mean()

    return loss, {
        "logit_mean":
            logits.detach().mean(),

        "policy_margin_mean":
            policy_margin.detach().mean(),

        "preference_accuracy": (
            (policy_margin - ref_margin) > 0
        ).float().mean().detach(),

        "chosen_reward_mean":
            (beta * (policy_chosen_logp - ref_chosen_logp)).detach().mean(),

        "rejected_reward_mean":
            (beta * (policy_rejected_logp - ref_rejected_logp)).detach().mean(),

        "dpo_margin_mean":
            (policy_margin - ref_margin).detach().mean(),
    }
