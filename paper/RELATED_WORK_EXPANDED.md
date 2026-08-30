# Expanded Related Work

This document provides an expanded related work section with 25+ citations organized by theme, including Nobel Prize-winning work.

---

## 2. Related Work

Grounded Commitment Learning draws on and contributes to several research traditions: incomplete contract theory from economics, multi-agent coordination from computer science, emergent communication from machine learning, and social scaling from anthropology.

### 2.1 Incomplete Contract Theory (Economics)

The foundational insight that contracts cannot specify all contingencies comes from the Nobel Prize-winning work of **Oliver Hart** and **Bengt Holmström** (2016 Nobel Prize in Economics).

**Hart and Moore (1988)** [1] established that incomplete contracts lead to hold-up problems: parties under-invest in relationship-specific assets because they fear exploitation when unspecified contingencies arise. Their model predicts that investment decreases with contract incompleteness—a prediction we validate empirically in Experiment 21.

**Hart (2017)** [2] extended this framework in his Nobel lecture, arguing that the allocation of control rights matters precisely because contracts are incomplete. GCL's failure-first specification can be understood as a mechanism for allocating control rights over failure contingencies.

**Grossman and Hart (1986)** [3] introduced the property rights approach, showing how ownership affects incentives when contracts are incomplete. GCL's stake mechanism provides analogous incentive alignment without requiring ownership transfer.

**Williamson (1985)** [4] developed transaction cost economics, emphasizing that bounded rationality and opportunism create contracting problems. GCL addresses bounded rationality through template abstraction and opportunism through verifiable commitments.

**Tirole (1999)** [5] surveyed incomplete contract theory, noting the tension between flexibility and commitment. GCL's failure-first approach resolves this by committing to failure handling while leaving success flexible.

### 2.2 Multi-Agent Coordination (Computer Science)

Multi-agent systems have developed various coordination mechanisms, each with distinct trade-offs.

**Smith (1980)** [6] introduced the Contract Net Protocol (CNP), enabling task allocation through manager-contractor negotiation. CNP assumes agents can interpret task announcements correctly—an assumption GCL relaxes through verifiable commitments.

**FIPA (2002)** [7] standardized agent communication with FIPA-ACL, providing performatives like INFORM, REQUEST, and COMMIT. While FIPA-ACL includes commitment performatives, it lacks GCL's failure-first specification and stake mechanisms.

**Wooldridge (2009)** [8] provided a comprehensive treatment of multi-agent systems, including coordination mechanisms. GCL extends this tradition by grounding coordination in verifiable behavioral contracts.

**Jennings (1993)** [9] proposed commitment-based coordination, arguing that commitments provide a principled basis for agent interaction. GCL operationalizes this insight with formal verification and failure handling.

**Castelfranchi (1995)** [10] developed a theory of social commitments, distinguishing between commitments to self and commitments to others. GCL focuses on inter-agent commitments with explicit verification.

**Singh (1999)** [11] formalized commitment protocols, providing semantics for commitment operations. GCL extends this with failure modes and stake-based incentives.

### 2.2.1 Contemporary Agent Contracts and LLM Coordination (2024-2026)

A rapidly growing body of work applies contract- and commitment-style formalisms to LLM-based agents; GCL must be positioned against it.

**Agent Contracts (arXiv:2601.08815, COINE @ AAMAS 2026)** formalizes resource-bounded contracts for autonomous AI systems, reporting large token-variance reductions and conservation guarantees in multi-agent delegation. It shares GCL's verifiability goal but focuses on resource bounds rather than failure-first specification, stake, or learned commitment selection.

**Self-Negotiated Contracts (arXiv:2607.22750, 2026)** algorithmically verifies whether negotiated agreements are structurally safe or beneficial for participants, using commitment-based abstractions. GCL complements this with a learning layer (which commitments to issue) and an empirical account of when decentralized selection beats central assignment.

**Self-Resource Allocation in Multi-Agent LLM Systems (arXiv:2504.02051)** compares LLMs as orchestrators vs. planners for task allocation, finding that explicit capability information improves allocation. Our Experiment 40 speaks directly to this design space: with symmetric observability, central assignment matches self-selection; the decisive variable is *relative* observation noise, plus an emergent-motivation channel unavailable to external assigners.

**Agentifying Agentic AI (arXiv:2511.17332, WMAC @ AAAI 2026)** observes that contemporary agentic systems lack explicit commitment mechanisms, with intentionality inferred from surface behavior. GCL provides precisely such an explicit mechanism, with verification and settlement semantics.

### 2.3 Multi-Agent Reinforcement Learning

Recent advances in MARL provide learning-based approaches to coordination.

**Foerster et al. (2016)** [12] demonstrated that agents can learn to communicate through reinforcement learning, developing emergent protocols. However, these protocols often lack interpretability and don't transfer across populations.

**Lowe et al. (2017)** [13] introduced MADDPG for multi-agent continuous control, showing that centralized training with decentralized execution enables coordination. GCL provides an alternative: decentralized training with commitment-based coordination.

**Rashid et al. (2018)** [14] developed QMIX for cooperative multi-agent learning, addressing credit assignment through value decomposition. GCL's stake mechanism provides explicit credit assignment through commitment fulfillment.

**Sunehag et al. (2018)** [15] proposed value decomposition networks (VDN), assuming additive value functions. GCL's sequential and parallel composition operators provide structured value decomposition.

**Yu et al. (2022)** [16] surveyed multi-agent reinforcement learning, identifying coordination as a key challenge. GCL addresses this through explicit commitment exchange rather than implicit policy coordination.

### 2.4 Emergent Communication

Research on emergent communication explores how agents develop shared protocols.

**Lazaridou et al. (2017)** [17] showed that neural agents can develop compositional communication through referential games. GCL's template hierarchy provides structured compositionality.

**Mordatch and Abbeel (2018)** [18] demonstrated emergence of grounded language in multi-agent settings. GCL grounds communication in behavioral consequences rather than referential semantics.

**Eccles et al. (2019)** [19] studied biases in emergent communication, finding that agents develop non-human-like protocols. GCL's failure-first specification enforces human-interpretable structure.

**Chaabouni et al. (2020)** [20] analyzed compositionality in emergent languages, finding it correlates with generalization. GCL's template composition provides explicit compositionality.

### 2.5 Social Scaling and Group Dynamics (Anthropology)

Research on human social organization informs our understanding of coordination limits.

**Dunbar (1992)** [21] proposed that neocortex size limits social group size, with humans limited to ~150 stable relationships. Our Experiment 23 finds analogous limits in GCL coordination (~100 agents).

**Dunbar (2010)** [22] extended this analysis to online social networks, finding similar constraints. This suggests coordination limits may be fundamental rather than biological.

**Hill and Dunbar (2003)** [23] examined social network size in hunter-gatherer societies, finding hierarchical structure at different scales. This motivates hierarchical GCL for large populations.

**Zhou et al. (2005)** [24] found discrete hierarchical layers in human social networks (5, 15, 50, 150). GCL populations may exhibit similar structure.

### 2.6 Mechanism Design and Game Theory

GCL connects to mechanism design through its incentive structures.

**Hurwicz (1960)** [25] founded mechanism design theory, asking how to design rules that achieve desired outcomes. GCL's stake mechanism can be viewed as a coordination mechanism.

**Myerson (1981)** [26] developed optimal auction theory, showing how to elicit truthful information. GCL's verification mechanism incentivizes truthful commitment-making.

**Maskin (2008)** [27] (Nobel Prize 2007) formalized implementation theory, characterizing when mechanisms can achieve social goals. GCL implements coordination through commitment verification.

**Roth (2002)** [28] (Nobel Prize 2012) developed matching theory, showing how to design stable matching mechanisms. GCL's commitment matching can be viewed as a form of stable matching.

### 2.7 AI Safety and Alignment

GCL contributes to AI safety through verifiable commitments.

**Amodei et al. (2016)** [29] identified concrete problems in AI safety, including safe exploration and distributional shift. GCL's failure-first specification addresses these through explicit failure handling.

**Christiano et al. (2017)** [30] proposed learning from human feedback (RLHF), aligning AI through preference learning. GCL provides complementary alignment through verifiable behavioral contracts.

**Irving et al. (2018)** [31] proposed AI safety via debate, using adversarial dynamics for alignment. GCL's verification mechanism enables similar adversarial checking.

**Kenton et al. (2021)** [32] surveyed alignment approaches, distinguishing between value learning and behavioral constraints. GCL provides behavioral constraints through commitments.

**Anthropic (2023)** [33] introduced Constitutional AI, training models to follow explicit principles. GCL's failure-first specification provides analogous explicit constraints.

### 2.8 Formal Verification and Contracts

GCL builds on formal methods for software contracts.

**Meyer (1992)** [34] introduced Design by Contract, specifying preconditions, postconditions, and invariants. GCL extends this with failure modes and inter-agent verification.

**Findler and Felleisen (2002)** [35] developed higher-order contracts for software, enabling compositional verification. GCL's composition operators provide analogous compositionality.

**Dimoulas et al. (2011)** [36] introduced correct blame for contracts, ensuring failures are attributed correctly. GCL's failure modes provide explicit blame attribution.

---

## References

### Incomplete Contract Theory (Nobel Prize Work)

[1] Hart, O., & Moore, J. (1988). Incomplete contracts and renegotiation. *Econometrica*, 56(4), 755-785.

[2] Hart, O. (2017). Incomplete contracts and control. *American Economic Review*, 107(7), 1731-1752. **[Nobel Lecture]**

[3] Grossman, S. J., & Hart, O. D. (1986). The costs and benefits of ownership: A theory of vertical and lateral integration. *Journal of Political Economy*, 94(4), 691-719.

[4] Williamson, O. E. (1985). *The Economic Institutions of Capitalism*. Free Press. **[Nobel Prize 2009]**

[5] Tirole, J. (1999). Incomplete contracts: Where do we stand? *Econometrica*, 67(4), 741-781. **[Nobel Prize 2014]**

### Multi-Agent Systems

[6] Smith, R. G. (1980). The contract net protocol: High-level communication and control in a distributed problem solver. *IEEE Transactions on Computers*, C-29(12), 1104-1113.

[7] FIPA. (2002). FIPA ACL Message Structure Specification. Foundation for Intelligent Physical Agents.

[8] Wooldridge, M. (2009). *An Introduction to MultiAgent Systems* (2nd ed.). Wiley.

[9] Jennings, N. R. (1993). Commitments and conventions: The foundation of coordination in multi-agent systems. *The Knowledge Engineering Review*, 8(3), 223-250.

[10] Castelfranchi, C. (1995). Commitments: From individual intentions to groups and organizations. In *Proceedings of ICMAS-95*, 41-48.

[11] Singh, M. P. (1999). An ontology for commitments in multiagent systems. *Artificial Intelligence and Law*, 7(1), 97-113.

### Multi-Agent Reinforcement Learning

[12] Foerster, J., Assael, Y. M., de Freitas, N., & Whiteson, S. (2016). Learning to communicate with deep multi-agent reinforcement learning. *NeurIPS*.

[13] Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017). Multi-agent actor-critic for mixed cooperative-competitive environments. *NeurIPS*.

[14] Rashid, T., Samvelyan, M., Schroeder, C., Farquhar, G., Foerster, J., & Whiteson, S. (2018). QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning. *ICML*.

[15] Sunehag, P., Lever, G., Gruslys, A., Czarnecki, W. M., Zambaldi, V., Jaderberg, M., ... & Graepel, T. (2018). Value-decomposition networks for cooperative multi-agent learning. *AAMAS*.

[16] Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Baez, A., & Wu, Y. (2022). The surprising effectiveness of PPO in cooperative multi-agent games. *NeurIPS*.

### Emergent Communication

[17] Lazaridou, A., Peysakhovich, A., & Baroni, M. (2017). Multi-agent cooperation and the emergence of (natural) language. *ICLR*.

[18] Mordatch, I., & Abbeel, P. (2018). Emergence of grounded compositional language in multi-agent populations. *AAAI*.

[19] Eccles, T., Bachrach, Y., Lever, G., Lazaridou, A., & Graepel, T. (2019). Biases for emergent communication in multi-agent reinforcement learning. *NeurIPS*.

[20] Chaabouni, R., Kharitonov, E., Bouchacourt, D., Dupoux, E., & Baroni, M. (2020). Compositionality and generalization in emergent languages. *ACL*.

### Social Scaling (Dunbar)

[21] Dunbar, R. I. (1992). Neocortex size as a constraint on group size in primates. *Journal of Human Evolution*, 22(6), 469-493.

[22] Dunbar, R. I. (2010). How many friends does one person need? Dunbar's number and other evolutionary quirks. *Faber & Faber*.

[23] Hill, R. A., & Dunbar, R. I. (2003). Social network size in humans. *Human Nature*, 14(1), 53-72.

[24] Zhou, W. X., Sornette, D., Hill, R. A., & Dunbar, R. I. (2005). Discrete hierarchical organization of social group sizes. *Proceedings of the Royal Society B*, 272(1561), 439-444.

### Mechanism Design (Nobel Prize Work)

[25] Hurwicz, L. (1960). Optimality and informational efficiency in resource allocation processes. In *Mathematical Methods in the Social Sciences*, 27-46. **[Nobel Prize 2007]**

[26] Myerson, R. B. (1981). Optimal auction design. *Mathematics of Operations Research*, 6(1), 58-73. **[Nobel Prize 2007]**

[27] Maskin, E. (2008). Mechanism design: How to implement social goals. *American Economic Review*, 98(3), 567-576. **[Nobel Lecture]**

[28] Roth, A. E. (2002). The economist as engineer: Game theory, experimentation, and computation as tools for design economics. *Econometrica*, 70(4), 1341-1378. **[Nobel Prize 2012]**

### AI Safety

[29] Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., & Mané, D. (2016). Concrete problems in AI safety. *arXiv preprint arXiv:1606.06565*.

[30] Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., & Amodei, D. (2017). Deep reinforcement learning from human preferences. *NeurIPS*.

[31] Irving, G., Christiano, P., & Amodei, D. (2018). AI safety via debate. *arXiv preprint arXiv:1805.00899*.

[32] Kenton, Z., Everitt, T., Weidinger, L., Gabriel, I., Mikulik, V., & Irving, G. (2021). Alignment of language agents. *arXiv preprint arXiv:2103.14659*.

[33] Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., ... & Kaplan, J. (2022). Constitutional AI: Harmlessness from AI feedback. *arXiv preprint arXiv:2212.08073*.

### Formal Methods

[34] Meyer, B. (1992). Applying "design by contract". *Computer*, 25(10), 40-51.

[35] Findler, R. B., & Felleisen, M. (2002). Contracts for higher-order functions. *ICFP*.

[36] Dimoulas, C., Findler, R. B., Flanagan, C., & Felleisen, M. (2011). Correct blame for contracts: No more scapegoating. *POPL*.

---

## Citation Summary

| Category | Count | Nobel Laureates |
|----------|-------|-----------------|
| Incomplete Contract Theory | 5 | Hart, Holmström, Williamson, Tirole |
| Multi-Agent Systems | 6 | - |
| Multi-Agent RL | 5 | - |
| Emergent Communication | 4 | - |
| Social Scaling | 4 | - |
| Mechanism Design | 4 | Hurwicz, Myerson, Maskin, Roth |
| AI Safety | 5 | - |
| Formal Methods | 3 | - |
| **Total** | **36** | **7 Nobel Laureates** |

---

## How GCL Relates to Each Tradition

| Tradition | GCL Contribution |
|-----------|------------------|
| Incomplete Contracts | Failure-first specification as partial completeness |
| Multi-Agent Systems | Verifiable commitments beyond message passing |
| MARL | Explicit coordination through commitment exchange |
| Emergent Communication | Structured templates vs emergent protocols |
| Social Scaling | Dunbar-like limits in AI coordination |
| Mechanism Design | Stake-based incentive alignment |
| AI Safety | Verifiable behavioral contracts for alignment |
| Formal Methods | Inter-agent contract verification |
