### Why does AI transparency matter?

Today’s most powerful AI systems shape how we search, communicate, create, and make decisions. Yet critical details about their development remain elusive.  
Who exactly developed these models? What kinds of data were they trained on? How much computational power did they require, and at what scale were they built?

These details are often undisclosed or buried in inconsistent technical papers and blog posts.  
This lack of visibility makes it difficult to assess risks, scrutinize claims, or understand who holds responsibility for the systems increasingly shaping society.  
Without transparency, trust is harder to build, oversight becomes limited, and public understanding falls behind.  
Measuring transparency is challenging—but increasingly necessary.

Several efforts have emerged to shed light on the opaque world of AI development.  
[Stanford CRFM](https://crfm.stanford.edu/)'s [*Foundation Model Transparency Index (FMTI)*](https://crfm.stanford.edu/fmti/) and the ODI’s [*AI Data Transparency Index (AIDTI)*](https://theodi.org/insights/reports/the-ai-data-transparency-index/) offer structured assessments across multiple dimensions, helping benchmark transparency across organizations.  
But these initiatives rely on manually intensive scoring processes, cover only a subset of models, and often don’t allow for user-led exploration.

As the number of models and developers continues to grow, there is a pressing need for more scalable, semi-automated tools—and ideally, a centralized database that tracks transparency and holds developers accountable for what they choose to share (or withhold).  
Beyond improving visibility, these systems could incentivize developers to adopt more transparent practices by highlighting gaps and making comparisons more public.

---
### A new tool for exploring AI transparency

With this visual explorer, we set out to spark progress toward more scalable and accessible transparency tools.  
We focused on creating a lightweight, interactive way to explore how transparent AI models are about key upstream inputs—specifically:

- Training dataset 
- Training dataset size (number of datapoints)
- Training Compute  
- Number of parameters  

This proof-of-concept draws from publicly available metadata compiled by [Epoch AI](https://epoch.ai/), using their curated list of [*Notable AI Models*](https://epoch.ai/data/notable-ai-models)—those considered historically significant, widely cited, or state-of-the-art—spanning from 1950 to mid-2025.

While the scope is limited, it allows us to surface meaningful patterns across some of the most influential models to date.  
Our hope is that this tool serves as a starting point: a small but concrete step toward more consistent transparency expectations, greater public visibility, and stronger accountability across the AI ecosystem.

---

### What this tool helps you do

While this is an experimental application—and some values are estimated or inferred—it offers a rare lens into the otherwise opaque development of AI systems.  
It doesn’t represent a complete inventory of disclosures, nor does it capture every nuance of transparency.  
Instead, it’s intended to help surface patterns and guide further inquiry on:

- **How developers approach transparency** - Identify which types of organizations disclose more (or less), and what kinds of upstream information are most frequently omitted

- **Compare organizational practices in upstream model information disclosure** - Explore how transparency varies across research labs, large tech companies, and public institutions, revealing distinct norms in how model inputs are shared.

- **Spot regional trends in transparency** - Assess how data-sharing practices vary by geographic region, and how local regulation, institutional culture, or governance priorities may shape openness.

- **Track changes in transparency over time** - Examine how transparency has evolved across model generations—uncovering whether disclosures are improving, stagnating, or becoming more selective as models scale.

- **Distinguish between “open access” model release formats and transparent practices** - Open access doesn’t always mean transparent—especially when disclosures lack consistency or completeness. This tool helps unpack what’s actually shared, beyond whether a model is simply downloadable, highlighting the gap between availability and true accountability.

---

The aim is not to rank or evaluate, but to illuminate current practices and help inform the path toward more open and responsible AI development.