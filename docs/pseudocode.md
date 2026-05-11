# Pseudocode for Key Algorithms

## 1. Example Selection (Top-K with Mahalanobis Distance)

```
Input:  query sentence q, training set D, encoder E, k
Output: top-k similar examples

1. Encode all training examples:
   for each d_i in D:
       v_i = E.encode(d_i.sentence + " [SEP] " + d_i.predicate + " [SEP]")

2. Compute inverse covariance matrix:
   V = stack all v_i
   Σ = covariance(V)
   Σ⁻¹ = pseudo_inverse(Σ)

3. Encode query:
   v_q = E.encode(q.sentence + " [SEP] " + q.predicate + " [SEP]")

4. Compute distances:
   for each v_i in database:
       diff = v_q - v_i
       dist_i = sqrt(diff^T · Σ⁻¹ · diff)   // Mahalanobis distance

5. Sort by distance (ascending) and return top-k
```

## 2. MMR (Maximal Marginal Relevance) Selection

```
Input:  query q, candidate pool C (from Top-K), k, λ
Output: k diverse examples

1. selected = []
2. remaining = C

3. for i = 1 to k:
   for each d in remaining:
       relevance = -distance(d, q)
       diversity = max_{s in selected} cosine_sim(d, s)
       mmr_score = λ · relevance - (1-λ) · diversity

   best = argmax(mmr_score)
   selected.append(best)
   remaining.remove(best)

4. return selected
```

## 3. ConE (Conditional Entropy) Ordering

```
Input:  test instances T, examples per instance (k=5), LLM M
Output: optimal ordering of k examples

1. Initialize scores: for each permutation π of {0,...,k-1}: score[π] = 0

2. For each test instance t in T:
   For each permutation π:
       prompt_full = system_prompt + examples[π] + query(t)
       prompt_ex   = system_prompt + examples[π]
       prompt_q    = system_prompt + query(t)

       CE_full = CrossEntropy(M, prompt_full) / len(prompt_full)
       CE_ex   = CrossEntropy(M, prompt_ex)   / len(prompt_ex)
       CE_q    = CrossEntropy(M, prompt_q)     / len(prompt_q)

       score[π] += (CE_full - CE_ex) / CE_q

3. Normalize: score[π] /= |T|

4. Return permutations sorted by score (ascending = best)
```

## 4. SRL Evaluation (Micro-F1)

### CoNLL Format (Position-based)
```
Input:  gold labels G, predicted labels P (aligned by position)
Output: Recall, Precision, F1

1. gold_count = count positions where G[i] contains "ARG" or "AUX"
2. pred_count = count positions where P[i] contains "ARG" or "AUX"
3. correct    = count positions where G[i] == P[i] and both are ARG/AUX

4. Recall    = correct / gold_count
5. Precision = correct / pred_count
6. F1 = 2 · R · P / (R + P)
```

### Dict Format (Pair-based)
```
Input:  gold pairs {(role, word)}, predicted pairs {(role, word)}
Output: Recall, Precision, F1

1. gold_count = |gold pairs|
2. pred_count = |pred pairs|
3. correct    = |gold pairs ∩ pred pairs|

4. Recall    = correct / gold_count
5. Precision = correct / pred_count
6. F1 = 2 · R · P / (R + P)
```
