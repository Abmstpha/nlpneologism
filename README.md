# Definition-to-Neologism Generation

This project is inspired by the research of my professor, Paul Lerner, as outlined in his paper [Towards Machine Translation of Scientific Neologisms](https://aclanthology.org/2024.jeptalnrecital-taln.17/). While the paper itself is in French, an abstract in English provides insight into its objectives:

> Scientific research continually discovers and invents new concepts, which are then referred to by new terms, neologisms, or neonyms in this context. As the vast majority of publications are written in English, disseminating this new knowledge in French often requires translating these terms, to avoid multiplying anglicisms that are less easily understood by the general public. We propose to explore this task using two thesauri, exploiting the definition of the term to translate it more accurately. To this end, we explore the capabilities of two large multilingual models, BLOOM and CroissantLLM, which can translate scientific terms to some extent. In particular, we show that they often use appropriate morphological procedures, but are limited by the segmentation into sub-lexical units. They are also biased by the frequency of term occurrences and surface similarities between English and French.

For my task, I am focusing on the "DEF" setting, which simplifies the problem as follows: given a definition, the goal is to generate the term that corresponds to it. I will evaluate the generated outputs using Exact Match, meaning the generated term must exactly match the reference.

For example:
- **Input**: "Having to do with the ability to transmit data in either direction."
- **Expected Output**: "bidirectional."

In this case, "bidirectional" is formed by prefixing "bi-" to "directional," itself derived by suffixing "-al" to "direction," which is present in the input definition. This project emphasizes understanding and modeling such morphological and linguistic patterns to achieve accurate term generation.
