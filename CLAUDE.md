# PPO with LLM-as-Judge for RLHF

## Project Overview

This project implements **Proximal Policy Optimization (PPO)** for fine-tuning a large language model (Zephyr 7B) using **LLM-as-Judge** instead of a traditional trained reward model.

### What We're Doing

Instead of the standard RLHF approach where you:
1. Collect preference data
2. Train a reward model
3. Use PPO with the reward model

We're doing:
1. Use existing prompts
2. **Skip training a reward model**
3. Use an LLM API (GPT-4, Claude, etc.) to score responses
4. Use PPO with the LLM judge scores

This is called **RLAIF (Reinforcement Learning from AI Feedback)** or **Direct RLAIF**.

## Why This Approach?

### Advantages:
- ✅ **No need to collect preference data** - Expensive and time-consuming
- ✅ **No need to train a reward model** - Saves compute and time
- ✅ **Better quality feedback** - LLMs like GPT-4/Claude are better judges than most reward models
- ✅ **Flexible** - Easy to change evaluation criteria by modifying the judge prompt

### Challenges:
- ❌ **API costs** - Each response needs to be scored by an API call
- ❌ **Slower** - API latency vs local reward model
- ❌ **Requires API access** - Need access to a capable LLM

## The Notebook: `3-RLHF.ipynb`

### Origin
This notebook is adapted from a GPT-2 PPO implementation. We've modified it to:
- Use **Zephyr 7B** instead of GPT-2
- Use **LLM judge** instead of trained reward model
- Use **our custom prompts** instead of sentiment analysis

### Key Components

#### 1. **Model Architecture** (Cells 7-9)
```
ModelForCausalLMWithValueHead
├── llm (Zephyr 7B) - The policy (actor) that generates text
└── v_head - Value function (critic) that estimates expected rewards
```

#### 2. **LLM Judge Function** (Cell 5)
```python
def get_llm_judge_score(prompt, response):
    # Call LLM API to score the response
    # Returns score in range [-1, 1]
```

**TODO:** Replace the placeholder with actual API call to GPT-4/Claude/Gemini

#### 3. **PPO Training Loop** (Cell 55)
For each batch:
1. **Generate**: Model produces responses to prompts
2. **Score**: LLM judge rates each response
3. **Reward**: Combine score with KL penalty (prevents model from diverging)
4. **Advantage**: Calculate how good each action was vs expected
5. **Update**: Use PPO to update the policy

#### 4. **Reward Function** (Cell 42)
```
reward = LLM_judge_score - β * KL_divergence
```

Where:
- `LLM_judge_score`: Quality rating from the LLM judge
- `β * KL_divergence`: Penalty for diverging from the original model
- `β = 0.2`: Controls how much we penalize divergence

## How to Use

### Setup (In Colab)

1. **Mount Google Drive** (Cell 1)
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Upload Dataset**
   - Upload `gpt_formatted_dataset_clean.csv` to your Google Drive
   - Update path in Cell 13 if needed

3. **Install Dependencies**
   - Transformers, datasets, bitsandbytes, etc.
   - Should auto-install when needed

### Running the Notebook

#### Phase 1: Test Components (Before Training)

Run these cells to verify everything works:

```
Cell 5:  Define LLM judge function (with placeholder)
Cell 8:  Define model architecture
Cell 9:  Load Zephyr 7B with value head
Cell 11: Load tokenizer
Cell 12: Create dummy dataset (for VS Code testing)
Cell 13: Load dataset
Cell 23: Tokenize prompts
Cell 31: Test single generation
Cell 34: Test LLM judge scoring
Cell 36: Test batch generation
```

**Expected behavior:** Model loads, generates text, scores get computed

#### Phase 2: Replace LLM Judge (Before Full Training)

In Cell 5, replace the placeholder:

```python
def get_llm_judge_score(prompt, response):
    # Example with OpenAI
    import openai

    judge_prompt = f"""Rate the quality of this response on a scale of 0-10.

Question: {prompt}
Answer: {response}

Consider:
- Accuracy and correctness
- Clarity and coherence
- Completeness

Rating (0-10):"""

    result = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.0,
    )

    # Extract score and normalize to [-1, 1]
    score = float(result.choices[0].message.content.strip())
    normalized_score = (score / 5.0) - 1.0  # 0-10 -> [-1, 1]

    return normalized_score
```

**Test this thoroughly before training!**

#### Phase 3: Full Training

Run the main training loop:

```
Cell 39: Create reference model (frozen copy)
Cell 55: Train RLHF (main training loop)
Cell 60: Validate on validation set
Cell 62: Save trained model
```

**Warning:** This will be slow with API calls. Start with a very small dataset!

## Hyperparameters (Tuned for Zephyr 7B)

### Model Settings
- `batch_size = 4` - Small batches for 7B model
- `mini_batch_size = 2` - Even smaller for memory
- `learning_rate = 1e-5` - Conservative for fine-tuning

### Generation Settings
- `output_min_length = 20` tokens
- `output_max_length = 100` tokens
- Adjust based on your prompts

### PPO Settings
- `ppo_epochs = 4` - How many times to reuse each batch
- `cliprange_ratio = 0.2` - PPO clipping parameter
- `beta = 0.2` - KL penalty coefficient (in `compute_rewards`)
- `gamma = 1.0` - Discount factor
- `lam = 0.95` - GAE lambda parameter

### When to Adjust:
- **Batch size too large?** → Reduce if OOM errors
- **Responses too short/long?** → Adjust `output_min/max_length`
- **Model changing too fast?** → Increase `beta` (more KL penalty)
- **Model not improving?** → Decrease `beta` or increase `learning_rate`

## Current Status

### ✅ Completed:
- [x] Replaced GPT-2 with Zephyr 7B
- [x] Added value head to model
- [x] Removed reward model dependency
- [x] Added LLM judge placeholder
- [x] Updated dataset to use custom prompts
- [x] Adjusted hyperparameters for 7B model
- [x] Added explanatory markdown cells

### 🚧 TODO:
- [ ] Implement actual LLM judge API call
- [ ] Test with real dataset (vs dummy data)
- [ ] Run full training loop end-to-end
- [ ] Validate that training improves responses
- [ ] Compare PPO vs DPO (separate project)

## Files

```
.
├── 3-RLHF.ipynb                          # Main notebook (this one)
├── ProximalPolicyOptimization_DirectRLAIF.ipynb  # Old TRL attempt (deprecated)
├── datasets/
│   └── gpt_formatted_dataset_clean.csv   # Prompts for training
├── ppo.py                                # TRL reference implementation
└── CLAUDE.md                             # This file
```

## Troubleshooting

### Memory Issues (OOM)
- Reduce `batch_size` (try 2 or 1)
- Reduce `mini_batch_size` (try 1)
- Reduce `output_max_length` (shorter responses)
- Consider using gradient checkpointing (add to model config)

### Model Not Improving
- Check LLM judge scores - are they reasonable?
- Is KL penalty too high? (try reducing `beta`)
- Are responses diverse? (check generation settings)
- Is learning rate too low? (try `3e-5`)

### Training Unstable (Loss exploding)
- Reduce learning rate
- Increase `beta` (more KL penalty)
- Check that `ratio_threshold = 10` is catching instability
- Verify LLM judge scores are in correct range `[-1, 1]`

### API Costs Too High
- Use a smaller dataset for testing
- Consider caching LLM judge scores
- Use a cheaper model for judge (GPT-3.5 vs GPT-4)
- Or... switch to training a reward model after all!

## Technical Details

### PPO Algorithm

1. **Collect Trajectories**: Generate responses with current policy
2. **Compute Rewards**: Score with LLM judge, subtract KL penalty
3. **Compute Advantages**: How good was each action vs expected?
4. **Update Policy**: Use clipped objective to prevent large updates

### Why PPO vs DPO?

**PPO (this notebook):**
- More flexible - can optimize for any reward
- Better for online learning (improving during training)
- More complex implementation
- Requires value function

**DPO (alternative):**
- Simpler - no reward model or value function
- Requires preference pairs (chosen vs rejected)
- More stable training
- Better for offline learning

## References

- **PPO Paper**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **RLHF**: [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325)
- **RLAIF**: [Constitutional AI](https://arxiv.org/abs/2212.08073)
- **Original Code**: Adapted from GPT-2 RLHF tutorial

## Questions?

Common questions and where to find answers:

**Q: How do I change the evaluation criteria?**
A: Modify the prompt in `get_llm_judge_score()` (Cell 5)

**Q: Can I use a different model than Zephyr?**
A: Yes! Change `model_name` in Cell 9. May need to adjust hyperparameters.

**Q: Why use PPO instead of supervised fine-tuning?**
A: PPO optimizes for a goal (high LLM judge scores), not just imitating examples.

**Q: How long does training take?**
A: Depends on dataset size and API speed. ~1-2 seconds per response with API calls.

**Q: Can I train without an API?**
A: Yes, but you'd need to train a reward model first, defeating the purpose of this approach.

---

*Last Updated: December 2025*
*For questions or issues, see the notebook comments or ask Claude!*
