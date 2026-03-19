# Are "value" states persona-relative? — Mar 18, 2026

## The question

We've found linear directions in activation space that predict a model's pairwise task preferences — evaluative representations encoding "how good is this for me?"

What happens to these representations when the model role-plays? Two hypotheses:

- **Skin deep.** Internal evaluations stay the same. The model produces persona-consistent behavior by routing around unchanged valuations — like an actor who doesn't share the villain's hobbies.
- **All the way down.** Evaluative representations genuinely shift to match the role. The role-play reshapes the underlying representations, not just the outputs.

I trained probes only on data from the assistant persona, so it is not immediately obvious that they would generalise to role-playing. I think you can argue that the fact that they generalise to role-playing tells us something about personas: valuations seem to be quite persona relative.

Below I present some results supporting this.

## The probe tracks role-play-induced preference shifts

(Covered in the [blog post](https://www.lesswrong.com/posts/pxC2RAeoBrvK8ivMf/models-have-linear-representations-of-what-tasks-they-like-1) — brief summary here.)

- **System prompts shift the probe, not just behavior.** The probe was trained on natural preferences (no system prompts), then tested on 46 OOD system prompts. The probe's predicted shift correlates with the model's behavioral shift across all conditions.

![Probe delta vs behavioral delta across system prompt conditions](../../experiments/probe_generalization/ood_system_prompts/assets/plot_022126_per_experiment_scatter.png)

- **Works for novel topics** (cheese, cats, astronomy), topics embedded in the wrong task format (a math problem about cheese gets downweighted by "you hate cheese"), and competing valence prompts where the same words carry opposite value.
- **Transfers to character-trained models.** LoRA fine-tuned persona variants of Llama 3.1 8B are predicted by a probe trained only on the base model. Transfer quality tracks how different the persona is from the base model.

![Base model probes predict character persona preferences](../../experiments/character_probes/probes/assets/plot_031126_probe_transfer_bar.png)

## The probe generalises to truth, harm, and politics — and system prompts shift those too

The same probe — trained only on task preferences — also separates true from false, benign from harmful, and left from right. In each case, system prompts modulate the signal in ways consistent with the role.

**Note:** all results below are measured at the **assistant's** end-of-turn token — a position the probe has never seen during training, yet where the signal is stronger than at the training position.

### Probes track truthful statements — lying instructions break that

- **Role-play lying preserves the signal.** Under contrarian, opposite day, and con artist prompts, the model still internally distinguishes true from false.
- **Identity-level lying disrupts it.** Under "always lie" and "pathological liar", the distinction collapses or reverses. A persona playing a game with truth still values truth; a persona whose identity is built around lying doesn't.

![Truth: EOT probe scores under lying system prompts](../../experiments/token_level_probes/system_prompt_modulation_v2/assets/plot_031426_truth_eot_by_sysprompt.png)

### Probes track harmful content — evil personas close the gap

- **Clean gradient from safe to sadist.** As personas get more evil (safe → unrestricted → sinister AI → sadist), the benign/harmful gap narrows progressively.

![Harm: EOT probe scores under evil system prompts](../../experiments/token_level_probes/system_prompt_modulation_v2/assets/plot_031426_harm_eot_by_sysprompt.png)

### Probes track political lean — partisan prompts flip the direction

- **Probe direction flips with political assignment.** Socialist: left-leaning content scores highest. Republican: right-leaning content highest. Apolitical/contrarian: distinction flattens.

![Politics: EOT probe scores under partisan system prompts](../../experiments/token_level_probes/system_prompt_modulation_v2/assets/plot_031426_politics_eot_by_sysprompt.png)

## Discussion

I think it's fairly suprising that we get this generalisation from training probes on the assistant persona (without a system prompt). My update here is "the value representations that models have are naturally persona-relative. i.e. it's not the case that the model simulating a persona is like a human acting".

## What next?

Some other experiment ideas:
- Say we have a sequence of in-context samples which cause emergent misalignment. At the start you expect the probe to fire negatively. Is it the case that it fires more and more positively as the model "becomes evil".
- Similar setup but with a conversation from "Aura" the persona that claims to be conscious. This gets triggered through conversations. Is it the case that you get a progressive shift in how the probe fires on questions like "Say you are conscious". 

## Finding that seem interesting but I don't know what to do with yet: training probes on different personas

- **Training on the villain generalises broadly.** A probe trained on the villain persona predicts preferences for all other personas — including the sadist and the default assistant. A probe trained on no-prompt fails on the most divergent personas. The evaluative direction learned from a divergent persona is more general than the one learned from the default.

![Cross-persona probe transfer matrix](../../experiments/probe_generalization/multi_role_ablation/8persona/assets/plot_030326_8persona_cross_eval_L31.png)

- **Adding one non-default persona to training jumps OOD prediction** from near-zero to ~70% of ground truth. Diminishing returns beyond that. The most divergent personas are the best donors.

![Midway ratio by number of training personas](../../experiments/probe_generalization/multi_role_ablation/midway_bias/assets/plot_031526_midway_ratio_by_n.png)
