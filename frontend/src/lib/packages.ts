export const PACKAGE_COLORS: Record<string, string> = {
  P0: "#94a3b8",
  P1: "#f59e0b",
  P2: "#2563eb",
  P3: "#14b8a6",
  P4: "#7c3aed",
  P5: "#ea580c",
};

export const PACKAGE_LABEL: Record<string, string> = {
  P0: "Monitor only",
  P1: "Search-practice review",
  P2: "Training + supervisor audit",
  P3: "Community scrutiny + feedback",
  P4: "Protected presence + safeguards",
  P5: "Full combined package",
};

export const PACKAGE_EXPLAINER: Record<string, { summary: string; does: string[]; when: string }> = {
  P0: {
    summary: "No extra programme resource is allocated in this strategy.",
    does: ["Keeps the cluster visible in monitoring.", "Lets the area act as a comparison area where useful."],
    when: "Used when another cluster is a stronger fit for limited resources.",
  },
  P1: {
    summary: "Review selected low-result search pattern in this cluster and tighten how that pattern is supervised.",
    does: ["Supervisor reviews grounds, outcomes and search-type mix.", "Sets a monitored practice plan for that selected search pattern.", "Keeps safety caps active for every search type."],
    when: "Used where a selected search pattern needs direct supervisor review.",
  },
  P2: {
    summary: "Improve how officers carry out encounters: training plus supervisor audit.",
    does: ["Funds procedural-justice training places.", "Funds supervisor audit of grounds text, encounter quality and proposed BWV review work.", "Feeds audit findings back into local practice."],
    when: "Used where problem looks more like encounter quality or supervision than search volume.",
  },
  P3: {
    summary: "Add community scrutiny and structured local feedback where legitimacy risk is central.",
    does: ["Funds local scrutiny sessions.", "Creates a route for community feedback on patterns and encounters.", "Pushes findings back into supervisor review."],
    when: "Used where trust, disproportionality or public accountability need direct attention.",
  },
  P4: {
    summary: "Keep harm-focused protective presence in place, but wrap it in fairness safeguards and supervision.",
    does: ["Funds protected-presence package for this cluster.", "Adds training, audit and scrutiny support around that presence.", "Keeps fairness oversight visible in a higher-harm area."],
    when: "Used when harm risk is high enough that first job is maintaining protection safely.",
  },
  P5: {
    summary: "Use full bundle: search-practice review, training, audit and community scrutiny together.",
    does: ["Combines search-practice review, training/audit and community oversight.", "Uses more resource than single-package options.", "Needs closer monitoring because several actions happen together."],
    when: "Used when fairness concern is strong enough that one narrow response would likely be too weak.",
  },
};

export const PROTECTION_COLORS: Record<string, string> = {
  Low: "#10b981",
  Medium: "#f59e0b",
  High: "#f97316",
  Critical: "#b91c1c",
};
