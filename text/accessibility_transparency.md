An AI model is considered “open” when it is made publicly available in a way that allows users to use, study, modify, and share critical components, including source code, model architecture and parameters, and ideally training data, under open-source licensing or equivalent transparent terms. Partial forms of openness (e.g., open-weight) may provide access to some components (e.g., trained weights), but do not necessarily guarantee full transparency or reproducibility (Ramlochan, 2023).

A common assumption is that open models are inherently transparent models. But does this hold?

Other frameworks, such as Stanford’s Foundation Model Transparency Index (FMTI), have found that open science models tend to demonstrate stronger transparency practices around development and deployment. To explore this relationship in the current dataset, transparency scores were analyzed across different model accessibility types—from fully closed to open weight releases.

A clear trend emerges:
- **API-only** show the lowest median transparency (1.0/4.0), with most scoring between 0 and 2. While widely deployed, they often share little about training data or compute. Rare exceptions, such as GPT-3 (davinci) and DALL-E, achieve full scores, but these are outliers, underscoring that limited access does not inherently preclude high documentation.

- **Hosted-access** models (no API) achieve the same median score of 1.0/4 but display more balanced variation. This may reflect benefits of external visibility, especially in public-sector deployments, though they still lack the depth of transparency seen in open-weight releases.

- Currently **unreleased** models tend to score higher (median 3.0/4.0), with many reporting parameters and training compute but omitting dataset details. However, it is not always clear whether these models are intended for future open or closed release.

- **Open-weight** models, whether unrestricted, non-commercial, or research-restricted, show the highest median transparency (3.0/4.0) and greater consistency in reporting key inputs. Models with restricted-use open weights (typically for research purposes) scored the highest average overall (3.05), though this reflects a smaller set of high-disclosure cases. Notably, some open-weight models (e.g., Qwen-Audio-Chat, Jumba 1.5-Large) scored only 1.0, highlighting that openness without a strong documentation ethos can still result in minimal transparency.

These results align with previous research in showing that open-weight models are, on average, more transparent. However, the presence of low-scoring open-weight models and high-scoring closed models demonstrates that access type alone is not a reliable proxy for transparency, and a robust documentation culture remains the decisive factor.
