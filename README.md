# MultiPromptDiffusion
Modification of sd 1.5 to take multiple prompts per cross attention layer

![grid_20250103-190405](https://github.com/user-attachments/assets/6832d79b-334f-4f4f-bf3c-e8c704611732)


model here: https://huggingface.co/CiaraRowles/MultiPromptDiffusion

A relatively simple modification of SD 1.5 to have a variable number of cross-attention layers per original layer in the arch, all layers past the first for each group are retrained to effectively be recursive.

Is this useful for anything? Not really, it doesn't improve the ability to understand prompts at all, but maybe someone will be doing something similar and want a baseline implementation, so here you go.

arch roughly based on the modified attention processors from: https://github.com/tencent-ailab/IP-Adapter/tree/main/ip_adapter
