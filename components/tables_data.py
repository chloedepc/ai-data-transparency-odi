model_id_vars = [
    {"Variable": "Model",
     "Description": "The best-known name of the model.",
     "Type and Source": "Text (original)"},
    {"Variable": "Developer Organization",
     "Description": "Entity/entities responsible for developing the model. Renamed for consistency.",
     "Type and Source": "Semi-structured text (transformed)"},
    {"Variable": "Publication Date",
     "Description": "The model's release, announcement, or publication date. If day/month information is missing, placeholder values are used (e.g., July 1st if month/day unknown).",
     "Type and Source": "Date (original)"},
    {"Variable": "Publication Year",
     "Description": "The model’s publication year, derived from Publication Date.",
     "Type and Source": "Numerical (transformed)"},
    {"Variable": "Country",
     "Description": "Country/countries associated with the developer organization(s). 'Multinational' indicates organizations associated with multiple countries.\n"
     "Country names were preserved but deduplicated.",
     "Type and Source": "Multi-label categorical (transformed)"},
    {"Variable": "Developer Region",
     "Description": "Grouped by World Bank analytical regions or marked as Cross-regional Collaboration. These categories were derived from the “Country” column in Epoch AI’s dataset, grouped into the World Bank’s analytical regions or classed as Cross-Regional Collaboration.",
     "Type and Source": "Categorical (engineered)"},
    {"Variable": "Developer Organization Type",
     "Description": (
         "Categories are based on organization classifications provided by Epoch AI. The typology was refined during early analysis by exploring the distribution of organization types across models. In cases involving multiple developers, “Industry–Academia Collaboration” was separated out as its own category due to its prominence, while other less frequent or mixed affiliations were grouped under “Cross-sector Collaboration.\n"
         "Final typology:\n"
         "– Industry: Private companies or corporate labs\n"
         "– Academia: University-affiliated institutions\n"
         "– Government / Public Sector: State-run or publicly funded institutions\n"
         "– Industry–Academia Collaboration: Formal partnerships between private companies and academic institutions\n"
         "– Cross-sector Collaboration: Projects spanning multiple sectors (e.g., public & private)\n"
         "– Research Collective: Decentralised / independent research communities\n"
         "– Unknown: No classification could be made based on available data."
     ),
     "Type and Source": "Categorical (transformed)"},
]

model_access_vars = [
    {"Variable": "Model Accessibility Type",
     "Description": (
        "How the public can interact with or access the model:\n"
        "– API Access: available only via an application programming interface (and potentially a hosted interface)\n"
        "– Hosted Access (No API): publicly usable via a hosted service, but without an API.\n"
        "– Unreleased: No public access\n"
        "– Open weights (unrestricted): Fully downloadable without restriction\n"
        "– Open weights (restricted use): Downloadable for specific use cases (e.g., research only)\n"
        "– Open weights (non-commercial): Downloadable for non-commercial purposes"
     ),
     "Type and Source": "Categorical (original)"},
]

app_specific_vars = [
    {"Variable": "Domain",
     "Description": "Broad area of application in Machine Learning (e.g. Language, 3D modeling, Medicine, Image generation).",
     "Type and Source": "Multi-label categorical (original)"},
    {"Variable": "Task",
     "Description": "Specific functions the model is designed to perform (e.g., Face recognition, Weather forecasting, Language modeling, etc.).",
     "Type and Source": "Semi-structured text (original)"},
]

finetune_vars = [
    {"Variable": "Base Model",
     "Description": "The original model used as a foundation for fine-tuning, if applicable.\n"
     "Missing values were relabelled as 'Unspecified'.",
     "Type and Source": "Semi-structured text (transformed)"},
    {"Variable": "Finetune compute",
     "Description": "Compute (measured in floating point operations) required for fine-tuning, if applicable.",
     "Type and Source": "Numerical (original)"},
]

core_transparency_vars = [
    {"Variable": "Training Compute",
     "Description": "Total compute (in floating-point operations) used for training. Derived via direct reports or estimated through GPU-hours/backpropagation steps.",
     "Type and Source": "Numerical (original)"},
    {"Variable": "Training Dataset",
     "Description": "Dataset used for model training. Standard datasets are often used, and can be selected as multiple choice options. Values that were either missing or explicitly labelled as “Unspecified unreleased” by Epoch AI were relabelled to “Unspecified” for simplicity.",
     "Type and Source": "Semi-structured text (transformed)"},
    {"Variable": "Training Dataset Size",
     "Description": "Number of datapoints in the training dataset, in the unit specified for a given task (e.g., images in image classification or number of words in language modeling).",
     "Type and Source": "Numerical (original)"},
    {"Variable": "Parameters",
     "Description": "Total learnable parameters in the model (for neural networks, these are the weights and biases).",
     "Type and Source": "Numerical (original)"},
    {"Variable": "Confidence (for numeric estimates)",
     "Description": (
        "Indicates how certain Epoch AI is about the accuracy of numerical estimates, particularly for training compute, which is often the most difficult to verif.\n"
        "Specify 90% confidence that the recorded values are within the following bounds:\n"
        "– Confident: ±3× (±0.5 orders of magnitude)\n"
        "– Likely: ±10× (±1 order)\n"
        "– Speculative: ±31× (±1.5 orders)\n"
        "Missing values relabelled as “Unknown”."
     ),
     "Type and Source": "Categorical (transformed)"},
]