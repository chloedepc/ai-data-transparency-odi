To help compare models more systematically, we assign each one a transparency score from 0 to 4, based on the presence of the following four inputs in publicly available metadata:

- **Training dataset** – What dataset was used to train the model?
    Knowing this helps assess the origin and potential risks of the data (e.g. copyright issues, bias, or quality concerns).

- **Training dataset size** – How many data points were in the training dataset? 
    This gives a sense of the model’s exposure to information, which can affect performance, generalization, and compute cost. 

- **Number of parameters** – How big is the model in terms of its internal structure?
    Parameter count is often used as a rough proxy for model complexity or capability

- **Training compute** – How much computational power was used to train the model?
    This metric relates to the model’s carbon footprint and financial cost, and helps distinguish between small-scale and industrial-scale development. 

Each model receives one point for every one of these variables that is reported in the Epoch AI dataset—and is therefore assumed to have been publicly disclosed. These four dimensions were selected because they are:
- Commonly reported (or inferred) in public model metadata
- Relevant to key questions around AI transparency and accountability–particularly around data provenance, model scale, and environmental impact.
- Available across enough models in the dataset to support meaningful comparison. 

The transparency score offers a simple, consistent lens to explore trends, gaps, and disclosure patterns across models. Its purpose is to provide a high-level, exploratory view of how often key information is made available. Each variable is equally weighted to keep the score interpretable and free from subjective judgment, helping surface patterns that may otherwise be difficult to detect across hundreds of models developed over time.

To ensure consistency in scoring and visualization, the analysis includes only models that appear to be trained from scratch. Fine-tuned models were excluded to avoid conflating pretraining transparency with that of later development stages, as their metadata often reflects only the fine-tuning process and omits the more opaque pretraining details. 

The score does not capture the full complexity of transparency, as it excludes factors such as labour and hardware use, social or environmental impacts, and other internal aspects of model development.