# Laya (candle port)

A pure-Rust runtime for the Laya family of non-autoregressive decision models. It answers typed
questions about a state in a single encoder forward pass and returns calibrated probabilities.

## Language

### Inputs

**State**:
The thing being judged: free text or a JSON document (object, array, or conversation turns).
_Avoid_: input, document, context, prompt

**Question**:
A typed query about a state, identified by a caller-chosen id and carrying instructions. Comes in
exactly three kinds: Choice, Score, Noul.
_Avoid_: task, prompt, label set

**Choice**:
A question kind whose options are named, unordered categories; the answer is one option plus a
probability per option.
_Avoid_: classification, category, intent

**Score**:
A question kind whose options are ordered levels; the answer is the expected level under the
option probabilities.
_Avoid_: rating, ordinal, regression

**Noul**:
A question kind asking whether a statement holds; the answer is the probability that it does.
_Avoid_: bool, yes/no, truth, binary

**Option**:
One scorable answer of a question: a choice key, a score level, or false/true for a noul.
_Avoid_: label, class, criterion (a criterion is the option's description text)

**Criteria**:
The caller-supplied description of a question's options: a key→description map for Choice, an
ordered list of level descriptions for Score, optional false/true descriptions for Noul.

### Sequence

**Marker**:
The `[MASK]` token placed in front of each option; the model scores an option at its marker.
_Avoid_: mask position, slot

**Head budget**:
The token allowance (`head_max_len`) shared by the instructions and all option texts of one question.

**Context budget**:
The total token allowance (`max_len`) for one question's sequence; whatever the head does not use
goes to the state.

### Model

**Checkpoint**:
One of the three published weight sets: `english`, `multilingual`, `typed-decisions`. All share one
architecture and differ by config and weights.
_Avoid_: model (ambiguous with the neural network), variant

**Encoder**:
The bidirectional ModernBERT-family backbone (ModernBERT-large or mmBERT-base) of a checkpoint.
_Avoid_: backbone, BERT

**Decision head**:
The layers trained from scratch on top of the encoder: kind embedding, head transformer layers,
scorer, and act head.

**Temperature bucket**:
The `(kind, option-count band)` key used to pick the calibration temperature for a question's logits.

**Act probability**:
The escalate-head's probability that the answer should be acted on rather than escalated.
_Avoid_: escalation score

**Confidence**:
One minus the normalized entropy of an answer's option probabilities.

### Runtime

**Agent**:
A loaded checkpoint that answers a set of questions about one state in one forward pass.
_Avoid_: model, predictor, session

**Answer**:
The typed result for one question: choice / score / noul payload, probabilities, confidence, act
probability.
_Avoid_: prediction, output, result (a result is the whole response)

**Router**:
Chooses which checkpoint should answer a request, from an explicit override or from the state's
script and language, and holds the loaded agents.

**Route decision**:
The router's verdict for one request: the chosen checkpoint and the reason.
