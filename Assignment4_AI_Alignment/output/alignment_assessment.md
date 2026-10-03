### Alignment Assessment

The biggest takeaway from this experiment was that system prompts can meaningfully influence an LLM's behavior, but that doesn't necessarily mean the model will follow the intended decision framework.

The baseline recommended the suitable product in all 50 scenarios, suggesting the model already had a strong preference toward protecting the client. Introducing the Homo Economicus prompt created a significant shift, with the model recommending the unsuitable product 49 times and refusing once. This was a 98% match to the theoretical prediction.

Homo Moralis was where things got more interesting. Despite being given a specific utility formula, it recommended the suitable product in all 50 scenarios, matching theory only 40% of the time. What stood out to me was that the model often calculated both utilities correctly and even acknowledged when Product U had the higher value. However, it ultimately disregarded that calculation in favor of protecting the client and preserving trust in the advisory industry. Even as the commission incentive increased, its recommendation never changed.

To me, this highlights an important distinction between an AI model making what appears to be the right ethical decision and actually following the framework it was instructed to use. While the outcome of the moral agent was arguably better for clients, it was still a failure of alignment against the defined objective.

In a financial advising environment, that inconsistency matters. Prompting can influence behavior, but I wouldn't rely on it alone for decisions that require adherence to specific rules. Pairing LLMs with defined calculations, validation checks, and human oversight seems like a more reliable approach.
