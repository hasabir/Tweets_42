# Tweets

- **1. Introduction to NLP**
    - What is NLP and its importance in processing textual data.
        - **Natural language processing (NLP) is a branch of [artificial intelligence](https://www.sas.com/en_nz/insights/analytics/what-is-artificial-intelligence.html) that helps computers understand, interpret and manipulate human language.**
        - Why is NLP importante to processing texual data
            - NLP makes it possible for computers to read text, hear speech, interpret it, measure sentiment and determine which parts are important.
            - **Structuring a highly unstructured data source**
    
    [What are AI fields](Wiki/What%20are%20AI%20fieldsmd)
    
- **2. Text Preprocessing Techniques**
    - **Tokenization**
        
        Tokenization is the process of breaking down text into smaller units, such as words or sentences. This is a crucial step in NLP as it transforms raw text into a structured format that can be further analyzed
        
    - **Stemming vs. Lemmatization**
        
        Stemming and lemmatization are two techniques in natural language processing (NLP) for simplifying words to their base or root forms. Although they both aim to reduce words to their root forms, they work in slightly different ways and have distinct purposes.
        
        ### 1. **Stemming**
        
        - **Definition**: Stemming is a rule-based process that removes word endings to get the base form, or "stem," of a word.
        - **Method**: It typically just chops off common suffixes without understanding the word’s meaning. For example, it might remove “-ing,” “-ed,” or “-s.”
        - **Example**: The words “running,” “runner,” and “ran” are all reduced to “run.”
        - **Usage**: Stemming is faster and simpler but can sometimes lead to awkward base forms (like “connect” and “connected” both being reduced to “connect”).
        
        ### 2. **Lemmatization**
        
        - **Definition**: Lemmatization is a more precise process that considers the context and meaning of a word, reducing it to its dictionary form or “lemma.”
        - **Method**: It uses vocabulary and grammatical rules to find the correct base form. Lemmatization identifies the part of speech (like noun, verb, etc.) to choose the appropriate lemma.
        - **Example**: The words “running” and “ran” would both become “run,” but the plural “mice” would become “mouse” since lemmatization recognizes these word variations.
        - **Usage**: Lemmatization is more accurate and context-aware but generally slower than stemming due to the additional analysis required.
        
        ### **Summary of Differences**
        
        | Aspect | Stemming | Lemmatization |
        | --- | --- | --- |
        | **Complexity** | Simple and rule-based | Complex, considers context and grammar |
        | **Output** | May produce incomplete or awkward forms | Produces correct dictionary forms |
        | **Speed** | Faster | Slower |
        | **Use Cases** | Quick text processing | Accurate text processing where meaning is essential |
        
        In short, stemming is quicker but less accurate, while lemmatization is slower but more accurate. The choice between them depends on the needs of your project: speed or accuracy.
        
    - **Regular Expressions (regexps):**
        - Most commonly called the search expression,
            
            they are used in phonology, morphology, text analysis,
            information extraction, & speech recognition.
            
            A regular expression is a set
            of characters that specify a pattern.
            
        - How can RE be used in NLP?
            1. Validate data fields (e.g., dates, email address, URLs,
            abbreviations)
            2. Filter text (e.g., spam, disallowed web sites)
            3. Identify particular strings in a text (e.g., token
            boundaries)
            4. Convert the output of one processing component into
            the format required for a second component
    - **Stop-words removal**
        
        Stop-words removal is a preprocessing technique in natural language processing (NLP) that involves removing common, non-informative words from text data. Stop-words are typically frequent words in a language that don’t add significant meaning to text analysis, such as “the,” “is,” “in,” and “and.” Here’s a structured breakdown:
        
        ### 1. **What Are Stop-Words?**
        
        - **Definition**: Stop-words are common words in a language that are often filtered out before text analysis because they carry little useful information for tasks like classification, clustering, or search.
        - **Examples**: In English, stop-words often include words like “the,” “a,” “and,” “of,” “to,” and “in.” These words occur frequently across documents but usually don’t contribute much to the meaning of the text.
        
        ### 2. **Why Remove Stop-Words?**
        
        - **Focus on Meaningful Words**: Removing stop-words helps focus on the more meaningful words, which are likely to contain the key information needed for analysis.
        - **Reduces Dimensionality**: By eliminating these common words, stop-words removal reduces the size of the vocabulary, leading to smaller and more efficient vector representations.
        - **Improves Model Efficiency**: Removing unnecessary words helps speed up the processing and training of NLP models by reducing the overall amount of data.
        
        ### 3. **How Stop-Words Removal Works**
        
        - **Predefined Lists**: Most NLP libraries (like NLTK, spaCy, or scikit-learn) have predefined lists of stop-words for different languages. When processing text, these words are simply removed based on these lists.
        - **Customization**: The list can be customized depending on the context. For example, words that are commonly found in business text may need to be removed in a business-specific analysis but retained in other contexts.
        
        ### 4. **Strengths and Limitations**
        
        - **Strengths**:
            - Simplifies text and improves processing speed.
            - Focuses analysis on more informative words.
        - **Limitations**:
            - Removing some words (like “not” or “but”) can change the meaning of the text, so careful selection of stop-words is important.
            - Stop-words removal may not be beneficial in tasks where every word carries information, like sentiment analysis.
        
        ### **Summary of Stop-Words Removal**
        
        | Aspect | Description |
        | --- | --- |
        | **Purpose** | Eliminates common, non-informative words from text |
        | **Focus** | Reduces text to more meaningful words for analysis |
        | **Efficiency** | Speeds up processing and reduces data size |
        | **Use Cases** | Used in most NLP tasks, especially classification and search engines |
        
        In summary, stop-words removal is a simple yet powerful step in NLP preprocessing. It enhances model performance by filtering out frequently occurring but non-essential words, though careful selection is required to avoid removing words that might hold contextual value.
        
    - **Handling misspellings (e.g., Levenshtein distance)**
        
        Handling misspellings in text data is an important preprocessing step in natural language processing (NLP), as it helps improve the quality and consistency of text for analysis. One popular method for identifying and correcting misspellings is using the Levenshtein distance. Here’s a structured explanation:
        
        ### 1. **What is Levenshtein Distance?**
        
        - **Definition**: The Levenshtein distance (also known as edit distance) is a metric that measures the minimum number of single-character edits (insertions, deletions, or substitutions) needed to change one word into another.
        - **Purpose**: It helps quantify how “close” two words are, which can be useful for identifying misspelled words and suggesting corrections based on known vocabulary words.
        
        ### 2. **How Levenshtein Distance Works**
        
        - **Calculation**: Given two words, the Levenshtein distance between them is the smallest number of edits required to turn one word into the other.
            - For example, the distance between “cat” and “bat” is 1 (substituting “c” for “b”).
            - For “kitten” and “sitting,” the distance is 3 (substitute “k” for “s,” substitute “e” for “i,” and add a “g”).
        - **Algorithm**: The Levenshtein distance can be calculated using dynamic programming, where a matrix is built to record the minimum edits needed to transform prefixes of one word into prefixes of another. This matrix allows efficient computation of edit distances for large texts.
        
        ### 3. **Using Levenshtein Distance to Handle Misspellings**
        
        - **Spell Correction**:
            1. For each word in the text, calculate the Levenshtein distance between the word and each word in a reference dictionary.
            2. If the distance is small (typically 1 or 2), suggest or automatically replace the word with the closest match in the dictionary.
        - **Approximate String Matching**: Levenshtein distance is also used in approximate matching when exact matches are unnecessary, such as in text searches where similar spellings are acceptable.
        
        ### 4. **Other Methods for Handling Misspellings**
        
        - **Soundex**: A phonetic algorithm that indexes words by their sound, making it useful for identifying words that may be misspelled but sound similar (e.g., “car” and “kar”).
        - **N-grams**: Compares overlapping sequences of characters within words to find similar patterns (e.g., “hello” and “hellp” share four-character sequences).
        - **Neural Spell Checkers**: Deep learning-based spell checkers can learn contextual spelling errors and are particularly useful in tasks where accuracy is critical.
        
        ### 5. **Strengths and Limitations**
        
        - **Strengths of Levenshtein Distance**:
            - Simple and effective for short words and single-word misspellings.
            - Helps quantify “closeness” between words for easy correction.
        - **Limitations of Levenshtein Distance**:
            - Computationally intensive for large text datasets.
            - Doesn’t consider word context, which can lead to incorrect substitutions in some cases (e.g., replacing “form” with “from” in certain contexts).
        
        ### **Summary of Levenshtein Distance for Misspellings**
        
        | Aspect | Description |
        | --- | --- |
        | **Purpose** | Measures closeness between words to identify misspellings |
        | **Method** | Calculates minimum edits (insertions, deletions, substitutions) |
        | **Output** | Integer value representing the similarity between two words |
        | **Use Cases** | Spell-checking, approximate search, text cleaning |
        
        In summary, Levenshtein distance is a powerful tool for identifying and correcting misspellings, especially when used alongside other techniques to handle large, complex text datasets. It’s particularly useful in cases where minor spelling errors need correction for analysis or retrieval tasks.
        
    - **N-grams (bigrams, trigrams)**
        
        N-grams are sequences of “N” words or tokens used together in natural language processing (NLP) to capture patterns or relationships between words in a text. They provide context beyond individual words by looking at word combinations, helping in tasks like text generation, search, and classification. Here’s a structured explanation:
        
        ### 1. **What Are N-grams?**
        
        - **Definition**: An N-gram is a continuous sequence of N items (usually words) in a text.
            - A **unigram** consists of single words, treating each word as an independent unit.
            - A **bigram** (N=2) consists of pairs of consecutive words.
            - A **trigram** (N=3) consists of triplets of consecutive words.
        - **Example**:
            - For the sentence “the cat sat on the mat”:
                - **Bigrams**: “the cat,” “cat sat,” “sat on,” “on the,” “the mat.”
                - **Trigrams**: “the cat sat,” “cat sat on,” “sat on the,” “on the mat.”
        
        ### 2. **Why Use N-grams?**
        
        - **Capture Context**: N-grams capture some of the local context in a text by looking at word sequences. This can help understand relationships between words that would be missed by focusing on single words alone.
        - **Feature Engineering for Text**: In tasks like text classification or sentiment analysis, N-grams can provide features that capture common phrases or word combinations, making it easier to identify patterns.
        - **Text Generation**: In language models, N-grams help predict the next word by considering previous words, enabling basic text generation.
        
        ### 3. **Common Applications of N-grams**
        
        - **Sentiment Analysis**: Bigram or trigram features can capture expressions like “not happy” or “very good” that might not be clear from individual words alone.
        - **Text Classification**: By using bigrams and trigrams, models can capture phrases specific to certain categories, such as “breaking news” in journalism.
        - **Speech Recognition**: N-grams help improve predictions by providing likely sequences of words, making it easier to recognize phrases correctly.
        
        ### 4. **Types of N-grams**
        
        - **Unigrams**: Single words (N=1); provide basic word-level frequency but no context.
        - **Bigrams**: Word pairs (N=2); capture simple relationships and help understand basic context (e.g., “New York”).
        - **Trigrams**: Three-word combinations (N=3); capture more nuanced context and commonly used phrases (e.g., “as soon as”).
        
        ### 5. **Strengths and Limitations**
        
        - **Strengths**:
            - Simple to implement and computationally efficient for smaller N.
            - Helps capture word relationships, improving context awareness in NLP tasks.
        - **Limitations**:
            - High Dimensionality: The number of possible N-grams grows exponentially with the vocabulary and value of N, creating sparsity and computational challenges.
            - Limited Context: N-grams capture only local context within a short word span, which may not be enough for complex dependencies over long sentences.
        
        ### **Summary of N-grams**
        
        | Aspect | Description |
        | --- | --- |
        | **Purpose** | Captures sequences of words for contextual analysis |
        | **Types** | Unigrams, bigrams, trigrams, etc., depending on the sequence length |
        | **Application** | Text classification, sentiment analysis, text generation |
        | **Limitations** | Increased dimensionality, limited context with higher N values |
        
        In summary, N-grams are a foundational technique for analyzing text, helping capture the relationships between words for improved accuracy in a variety of NLP tasks. They provide context beyond individual words, making them valuable for text-based models and analysis.
        
    - **Collocations and capturing multi-word phrases**
        
        Collocations are combinations of words that frequently appear together and convey a specific meaning. These word pairs or groups have a stronger association than would be expected by random chance, making them important in natural language processing (NLP) for understanding language structure and semantics. Capturing collocations and multi-word phrases can enhance NLP models by providing insights into common expressions, idioms, and phrases that convey meaning beyond individual words.
        
        Here’s a structured explanation:
        
        ### 1. **What Are Collocations?**
        
        - **Definition**: Collocations are word combinations that occur more frequently together than by random chance, forming natural, meaningful phrases.
        - **Examples**:
            - **Bigrams**: "fast food," "strong tea," "make a decision"
            - **Trigrams**: "as soon as," "go hand in hand," "give it up"
        
        ### 2. **Types of Collocations**
        
        - **Grammatical Collocations**: Phrases made up of specific grammatical structures, such as verb-noun or adjective-noun pairs.
            - Example: “take a look,” “commit a crime,” “heavy rain.”
        - **Lexical Collocations**: Phrases where words are associated based on language usage rather than grammar.
            - Example: “make progress,” “highly likely,” “free speech.”
        
        ### 3. **Why Capture Collocations and Multi-Word Phrases?**
        
        - **Improved Text Understanding**: Collocations help capture the natural way words are combined, which improves understanding of meaning, especially in idiomatic phrases like “break the ice.”
        - **Enhanced Language Models**: Including collocations in language models can improve prediction and classification by capturing expressions that have unique meanings.
        - **Context Clarity**: Capturing phrases like “strong tea” vs. “strong argument” can clarify meaning that would be missed by analyzing single words alone.
        
        ### 4. **Methods for Identifying Collocations**
        
        - **Frequency-Based Methods**: Count how often words appear together. Higher frequency combinations are likely to be collocations, but this may still include irrelevant pairs.
        - **Statistical Measures**:
            - **Pointwise Mutual Information (PMI)**: Measures how much more often two words appear together than by chance. Higher PMI scores indicate a stronger association.
            - **Chi-Square Test**: A statistical test that assesses whether the occurrence of two words together is independent or not.
            - **Likelihood Ratio Test**: Used to evaluate the likelihood of a collocation based on statistical evidence.
        
        ### 5. **Applications of Collocations and Multi-Word Phrases**
        
        - **Information Retrieval**: Collocations help improve search results by recognizing phrases as single units, improving accuracy when searching for common expressions.
        - **Sentiment Analysis**: Multi-word phrases can capture expressions with emotional weight (e.g., “highly recommend” or “not good enough”).
        - **Machine Translation**: Collocations ensure accurate translation by keeping phrases intact instead of translating each word separately.
        
        ### 6. **Strengths and Limitations**
        
        - **Strengths**:
            - Helps capture natural language use and idiomatic expressions.
            - Improves model accuracy by focusing on meaningful word pairs or groups.
        - **Limitations**:
            - Data-Intensive: Requires large text corpora to accurately identify collocations.
            - May Capture Irrelevant Pairs: High-frequency word pairs may still include irrelevant or coincidental combinations.
        
        ### **Summary of Collocations and Multi-Word Phrases**
        
        | Aspect | Description |
        | --- | --- |
        | **Purpose** | Identifies natural, frequently occurring word pairs or groups |
        | **Types** | Grammatical and lexical collocations |
        | **Techniques** | Frequency counts, PMI, chi-square test, likelihood ratio |
        | **Applications** | Search, sentiment analysis, translation, text generation |
        | **Limitations** | Requires large datasets, may capture irrelevant pairs |
        
        In summary, collocations and multi-word phrases help capture natural language patterns, providing essential context for NLP tasks. These phrases often convey specific meanings that single words alone cannot, making them crucial for accurate language understanding and processing.
        
    
- **3. Text Vectorization techniques**
    - **TF-IDF (Term Frequency-Inverse Document Frequency)**
        
        Term Frequency-Inverse Document Frequency (TF-IDF) is a more advanced technique in natural language processing (NLP) that represents text by considering both the importance of a word in a document and how unique that word is across a set of documents. Here’s a structured explanation:
        
        ### 1. **How TF-IDF Works**
        
        - **Definition**: TF-IDF combines two metrics—*Term Frequency (TF)* and *Inverse Document Frequency (IDF)*—to determine the importance of a word within a document relative to its occurrence in a collection of documents (or corpus).
        - **Formula**:
            - **Term Frequency (TF)**: Measures how often a word appears in a document. It’s calculated as the frequency of the term in a specific document divided by the total number of terms in that document.
            
            `tf(t,d) = count of t in d / number of words in d`
            - **Inverse Document Frequency (IDF)**: Measures how common or rare a word is across all documents in the corpus. It’s calculated as the logarithm of the total number of documents divided by the number of documents containing the term.
            
            ```python
            idf(t,D) = log(N / df(t))
            
            where:
            N = total number of documents
            df(t) = number of documents containing term t
            ```
            
            This formula gives higher weight to rare terms (appearing in few documents) and lower weight to common terms (appearing in many documents). For example:
            
            - If a term appears in all documents, df(t) = N, so idf = log(1) = 0
            - If a term appears in only 1 document, df(t) = 1, so idf = log(N), giving it the highest weight
            - Sometimes a +1 is added to prevent division by zero: log(N / (df(t) + 1))
            - **TF-IDF Score**: The TF-IDF score is then calculated by multiplying TF and IDF for each term in each document:
            
            <aside>
            💡
            
            TF-IDF(t,d,D)=TF(t,d)×IDF(t,D)
            
            </aside>
            
        
        ### 2. **Purpose and Advantages**
        
        - **Emphasis on Unique Terms**: Words that appear frequently within a document (high TF) but are rare across the corpus (high IDF) will have higher TF-IDF scores, marking them as more significant.
        - **Reduces Common Words' Influence**: Common words (e.g., “the,” “and”) that appear across many documents have low IDF values, reducing their TF-IDF score. This helps in focusing on words more unique to each document.
        
        ### 3. **Characteristics of TF-IDF**
        
        - **No Context or Order**: Like Bag of Words, TF-IDF doesn’t consider word order or contextual meaning. It focuses purely on frequency and rarity.
        - **Sparse Representation**: Since TF-IDF vectors are based on a large vocabulary, the resulting matrix can be sparse, with many zero values.
        - **Useful for Information Retrieval**: TF-IDF is widely used in search engines to rank documents based on keyword relevance, as well as in text classification.
        
        ### 4. **Strengths and Limitations**
        
        - **Strengths**:
            - Gives higher importance to unique, relevant terms.
            - Reduces the influence of common words.
        - **Limitations**:
            - Ignores the sequence and relationships between words.
            - Can still be computationally intensive for large datasets.
        
        ### **Summary of TF-IDF**
        
        | Aspect | Description |
        | --- | --- |
        | **Purpose** | Weighs the importance of words based on frequency and rarity |
        | **Order Sensitivity** | Ignores order; focuses only on individual word significance |
        | **Complexity** | More complex than Bag of Words; requires computation of IDF |
        | **Output** | Sparse vectors with weighted scores for each word |
        | **Use Cases** | Information retrieval, document ranking, keyword extraction |
        
        In summary, TF-IDF is a valuable technique for finding significant words in a document by balancing frequency and rarity, making it highly effective for text mining and search applications where understanding word importance is crucial.
        
    - **Bag of Words (BoW)**
        
        Bag of Words (BoW) is a technique in natural language processing (NLP) used to represent text data in a way that machines can understand. It is a straightforward model that focuses on the words present in the text without considering their order or context. Here’s a structured breakdown:
        
        ### 1. **How Bag of Words Works**
        
        - **Definition**: BoW creates a vocabulary from all unique words in a set of documents and represents each document as a “bag” of these words, regardless of their order.
        - **Method**:
            1. First, a list of all unique words (the vocabulary) across all documents is created.
            2. Each document is then represented as a vector of word counts, where each position in the vector corresponds to a word in the vocabulary, with the value indicating the frequency of that word in the document.
        - **Example**: Suppose we have two documents:
            - Doc 1: “The cat sat on the mat”
            - Doc 2: “The dog lay on the mat”
            - BoW will identify unique words across both (like "the," "cat," "sat," "dog," etc.) and represent each document by the count of these words. For example, "the" would appear twice in each document, while "cat" and "dog" would appear only in their respective documents.
        
        ### 2. **Characteristics of Bag of Words**
        
        - **No Context**: BoW disregards word order and sentence structure, so words are treated independently. For example, “cat sat” and “sat cat” are represented the same way.
        - **Simplicity**: BoW is a simple model that only needs basic word counts, making it computationally efficient.
        - **Sparse Representation**: Since each document is represented by counts of words from the full vocabulary, BoW often results in sparse vectors (many zero values), especially for large vocabularies.
        
        ### 3. **Strengths and Limitations**
        
        - **Strengths**:
            - Easy to implement and understand.
            - Effective for tasks where word frequency is more important than order (like spam detection).
        - **Limitations**:
            - Ignores word order, which can lose important context.
            - Can become computationally heavy with large vocabularies, leading to high-dimensional vectors.
        
        ### **Summary of Bag of Words (BoW)**
        
        | Aspect | Description |
        | --- | --- |
        | **Purpose** | Represents text data by word frequency |
        | **Order Sensitivity** | Ignores the order of words (focuses on presence and frequency) |
        | **Complexity** | Simple and easy to implement |
        | **Output** | Sparse vectors with word counts |
        | **Use Cases** | Basic text classification, spam filtering, etc. |
        
        In summary, Bag of Words is a simple and fast way to represent text, especially effective for tasks focused on word frequency. However, it lacks context-awareness, so it may not be ideal for tasks that rely on understanding word relationships.
        
    - **0 or 1 vectorization**
        
        In Natural Language Processing (NLP), **0 or 1 vectorization** typically refers to **binary encoding** or **binary vectorization**. This technique involves representing words, phrases, or documents as binary vectors where each element of the vector corresponds to the presence or absence of a particular feature, such as a word in the vocabulary. In this method:
        
        - **1** represents the presence of a feature (e.g., a word).
        - **0** represents the absence of that feature.
        
        One common approach for binary vectorization is **Binary Term Frequency (TF)**, where each word in a document is represented by a 1 if it appears, or 0 if it doesn't.
        
        Another example is **One-Hot Encoding**, where each word in a vocabulary is represented as a unique binary vector. If a word exists in a document, its corresponding index in the vector is set to 1, and all other indices are set to 0.
        
        ### Example:
        
        For a vocabulary of words: `["apple", "banana", "cherry"]`
        
        - Document 1: "apple cherry" → `[1, 0, 1]` (apple and cherry are present)
        - Document 2: "banana" → `[0, 1, 0]` (only banana is present)
        
        Binary encoding is simple but can lead to sparse vectors, especially with large vocabularies, and it doesn't capture the semantic relationships between words. For more nuanced representation, techniques like TF-IDF or Word2Vec are often preferred.
        
- **3. Similarity Measurement**
    - **Cosine Similarity**
        - Cosine similarity is a metric used to measure the similarity between two vectors by calculating the cosine of the angle between them.
        - The formula for cosine similarity is:
            
            ```python
            (x, y) = x . y / ||x|| ×× ||y||
            
            where x and y are vectors
            ```
            
        - Key characteristics:
            - Values range from -1 to 1, where 1 means identical direction, 0 means orthogonal, and -1 means opposite direction
            - Independent of vector magnitude, focusing only on orientation
            - Particularly useful for high-dimensional spaces like text analysis
        - Common applications:
            - Document similarity comparison
            - Text classification
            - Information retrieval
            - Recommendation systems
    - Identifying the most similar text pairs within a dataset.
- **4. Machine Learning for NLP**
    - **Logistic Regression**
        
        ## Description
        
        Logistic Regression is a statistical method used for binary 
        classification problems. It predicts the probability that a given input 
        belongs to a particular category (e.g., 0 or 1, yes or no) using a 
        logistic function.
        
        ## Example Representation
        
        For instance, in a medical diagnosis scenario, Logistic Regression
         could be used to predict whether a patient has a disease (1) or not (0)
         based on features like age, blood pressure, and cholesterol levels.
        
        ## Formula
        
        The logistic function is represented as:
        
        $$
        p(X) = \frac{1}{1 + e^{-(b_0 + b_1X_1 + b_2X_2 + ... + b_nX_n)}}
        $$
        
        Where:
        
        - p(X)p(X) is the predicted probability of the positive class.
        - b0b0​ is the intercept.
        - b1,b2,...,bnb1​,b2​,...,bn​ are the coefficients for each feature X1​,X2​,...,Xn​.
            
            X1,X2,...,Xn
            
    - **Multinomial Naive Bayes (MultinomialNB)**
        
        ## Description
        
        Multinomial Naive Bayes is a variant of the Naive Bayes classifier
         that is particularly suited for classification with discrete features 
        (e.g., word counts for text classification). It assumes that the 
        features follow a multinomial distribution.
        
        ## Example Representation
        
        In text classification, such as spam detection, MultinomialNB can 
        be used to classify emails as "spam" or "not spam" based on the 
        frequency of words in the email content.
        
        ## Formula
        
        The probability of class
        
        Ck
        
        Ck​ given a feature vector
        
        X
        
        X is calculated as:
        
        $$
        P(C_k|X) = \frac{P(X|C_k)P(C_k)}{P(X)}
        $$
        
        For multinomial distributions:
        
        $$
        P(X|C_k) = \frac{n_k!}{n_{k1}!n_{k2}!...n_{km}!} \prod_{j=1}^m P(x_j|C_k)^{n_{kj}}
        $$
        
        Where:
        
        - nknk​ is the total count of features in class Ck​.
            
            Ck
            
        - nkjnkj​ is the count of feature j in class Ck​.
            
            j
            
            Ck
            
        - P(Ck)P(Ck​) is the prior probability of class Ck​.
            
            Ck
            
    - **Support Vector Classification (SVC)**
        
        ## Description
        
        Support Vector Classification is a supervised learning model that 
        analyzes data for classification and regression analysis. It constructs 
        hyperplanes in a high-dimensional space to separate different classes.
        
        ## Example Representation
        
        In image classification, SVC can be used to distinguish between 
        different types of objects (e.g., cats vs. dogs) by finding an optimal 
        hyperplane that separates these classes based on their features.
        
        ## Formula
        
        The decision function for SVC can be expressed as:
        
        $$
        f(x) = w^Tx + b
        $$
        
        Where:
        
        - w is the weight vector (normal to the hyperplane).
        - x is the input feature vector.
        - b is the bias term.
            
            The goal is to maximize the margin between different classes while minimizing classification errors.
            
    
- **6. Word2Vec (Bonus)**
    
    [Word2Vec : Natural Language Processing](https://www.youtube.com/watch?v=f7o8aDNxf7k)
    
    [Coding Word2Vec : Natural Language Processing](https://www.youtube.com/watch?v=d2E-pU4H2gc&t=9s)