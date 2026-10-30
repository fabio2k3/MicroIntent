# Few-Shot Query Intent Detection via

# Relation-Aware Prompt Learning

## Liang Zhang, Yuan Li, Shijie Zhang, Zheng Zhang, and Xitong Li

```
Abstract —Intent detection is a crucial component of modern conversational systems, since accurately identifying user intent at the
beginning of a conversation is essential for generating effective responses. Currently, most of the intent detectors can only work
effectively with the assumption that high-quality labeled data is available (i.e., the collected data is labeled by domain experts). To ease
this process, recent efforts have focused on studying this problem under a more challenging few-shot scenario. These approaches
primarily leverage large-scale unlabeled dialogue text corpora to pretrain language models through various pretext tasks, followed by
fine-tuning for intent detection with very limited annotations. Despite the improvements achieved, existing methods have predominantly
focused on textual data, neglecting to effectively capture the crucial structural information inherent in conversational systems, such as
the query-query relation and query-answer relation. Specifically, the query-query relation captures the semantic relevance between two
queries within the same session, reflecting the user’s refinement of her request, while the query-answer relation represents the
conversational agent’s clarification and response to a user query. To address this gap, we propose SAID, a novel framework that
integrates both textual and relational structure information in a unified manner for model pretraining for the first time. Firstly, we
introduce a relation-aware prompt module, which employs learnable relation tokens as soft prompts, enabling the model to learn shared
knowledge across multiple relations and become explicitly aware of how to interpret query text within the context of these relations.
Secondly, we reformulate the few-shot intent detection problem using prompt learning by creating a new intent-specific relation-aware
prompt, which incorporates intent-specific relation tokens alongside the semantic information embedded in intent names, helping the
pretrained model effectively transfer the pretrained knowledge acquired from related relational perspectives. Building on this
framework, we further propose a novel mechanism, the query-adaptive attention network (QueryAdapt), which operates at the relation
token level by generating intent-specific relation tokens from well-learned query-query and query-answer relations explicitly, enabling
more fine-grained knowledge transfer. Extensive experimental results on two real-world datasets demonstrate that SAID significantly
outperforms state-of-the-art methods, achieving improvements of up to 27% in the 3-shot setting. When equipped with the relation
token–level QueryAdapt module, it yields additional performance gains of up to 21% in the same setting. Experimental results also
highlight the flexibility of both SAID and SAID (+QueryAdapt) as plug-and-play solutions compatible with various backbone models.
```
```
Index Terms —Few-shot Intent Detection, Pretrained Language Model, Prompt Learning
```
## ✦

## 1 INTRODUCTION

```
Task-oriented conversational systems enable users to inter-
act with computer applications through multi-turn natu-
ral language interactions to accomplish specific tasks with
well-defined semantics. With recent advancements in large
language models (LLMs) [1], [2], [3], [4], we have taken a
step further and witnessed the emergence of modern open-
domain conversational systems, known as LLM-powered
chatbots such as ChatGPT [5], [6], [7]. In this context,
intent detection, the task of classifying a user’s behaviors
(i.e., queries within each conversation) into one of several
mutually exclusive intent classes, becomes a cornerstone
as accurately identifying user intent at the beginning of
a conversation is crucial for downstream tasks such as
effective response generation [8] and successful instruction
execution [9].
```
- L. Zhang is with the Hong Kong University of Science and Technology
    (Guangzhou), China. Email: liangzhang@hkust-gz.edu.cn.
- Y. Li and S. Zhang and X. Li are with Shenzhen MSU-BIT University,
    China. Email: roygodly123@gmail.com, zhang.shijie1101@gmail.com,
    lix@hec.fr.
- Z. Zhang is with the Harbin Institute of Technology, China. Email:
    zhengzhang@hit.edu.cn.
- Corresponding author: Xitong Li and Shijie Zhang

```
However, to boost the performance of intent detection,
costly and time-consuming manual annotation efforts are
required. Additionally, user intents evolve over time, mak-
ing it difficult to collect sufficient labeled data in a timely
manner to cope with these dynamic intents in real-world
commercial systems. These challenges highlight the need
for sample-efficient methods that facilitate effective intent
detection in a low-data scenario where only several labeled
examples are available per intent (i.e., the so-called few-
shot learning setups). Recently, this problem has garnered
increasing research attention [10], [11], [12], [13].
To address the few-shot intent detection problem, recent
works have primarily leveraged large-scale pretrained lan-
guage models (PLMs), such as BERT [1] and RoBERTa [14].
These models are usually pretrained on large-scale dialogue
text corpora and then fine-tuned for intent detection utiliz-
ing different pretraining tasks. For example, IntentBERT [10]
introduces supervised pretraining, which simultaneously
optimizes a language modeling loss and a supervised loss
on a small labeled dataset through a standard intent clas-
sification task. Shnarch et al. [11] also adopt a classification
task on top of BERT; however, instead of relying on man-
ually labeled intent dataset, they generated pseudo-labels
through unsupervised clustering of query texts. LLM-Aug
[15] further generates pseudo-labeled query texts by lever-
```
## arXiv:2509.05635v1 [cs.CL] 6 Sep 2025


aging powerful large language models. In addition to task-
oriented supervised pretraining, contrastive learning-based
pretraining with data augmentation is also often employed
with PLMs to tackle the few-shot intent detection problem.
Specifically, CPFT [13] introduces a masked text augmenta-
tion strategy, treating a user query and its masked version
as positive samples, and then pretrains a BERT encoder
by bringing their representations closer through contrastive
learning. Zhang et al. [12] adopt a similar approach by
introducing contrastive loss as a regularization technique
to optimize the query representation space. However, they
rely on a different data augmentation strategy borrowed
from SimCSE [16], where the same query text with different
dropout ratios is treated as positive samples.
Nevertheless, most existing methods have primarily fo-
cused on pretraining PLMs based solely on the query text,
overlooking the important structural information inherent
in these conversational systems. Consider a typical example
of a user interacting with an LLM-powered chatbot, as
illustrated in Figure 1. In this scenario, Mary initiates a
multi-turn query session related to the topic of “climate
change”, and the chatbot generates a unified textual re-
sponse for each specific query accordingly. From this ex-
ample, two critical types of structural information can be
identified: (1) the query-query relation, which refers to the
semantic relevance between two queries within the same
session and reflects the user’s refinement of her request,
formed due to the continuity of user interaction behaviors;
and (2) the query-answer relation, which reflects the LLM’s
clarification of a user query, emerging naturally during the
interaction between the user and the chatbot. Such structural
information enhances the understanding of a given query
from various relational perspectives and provides valuable
insights for intent detection. Specifically, intent detection
can be viewed as interpreting an incoming new query
through the lens of intent name, which is essentially another
relational perspective to clarify the query text and is closely
related to both query-query and query-answer relations. If
a pretrained model can learn to comprehend a user query
from these typical relational perspectives already embedded
in large-scale dialogue text corpora, it can generalize well
and transfer this pretrained knowledge effectively, thereby
facilitating the downstream few-shot intent detection task.
Motivated by this, we propose a new framework called
SAID ( **S** tructure- **A** ware pretraining for few-shot query
**I** ntent **D** etection) in this paper, which unifies both textual
and relational structure information inherent in conversa-
tional systems for model pretraining. The key idea is to
introduce a relation-aware prompt module, which incorpo-
rates a few learnable relation tokens as special soft prompts,
positioned between the original text tokens of a pair of
queries or between a query and its answer. This module is
highly flexible and enables the model to handle multiple
relations jointly within a single framework. The PLM is
then pretrained using the relation-aware prompts with a
structure-aware masked language modeling task, encourag-
ing it to learn shared knowledge across different relations
and become explicitly aware of how to interpret query
text within the context of these relations. When applied
to the downstream few-shot intent detection task, intent-
specific relation tokens are further introduced alongside the

```
semantic information embedded in intent names, creating
a new intent-specific relation-aware prompt that helps the
model to effectively transfer the knowledge acquired from
related relational perspectives during pretraining stage.
In addition to the task reformulation via intent-specific
relation-aware prompt at the modeling paradigm level, we
further introduce a novel mechanism named QueryAdapt
( Query - Adapt ive attention network) and propose an en-
hanced model named SAID (+QueryAdapt), to explicitly
transfer relational knowledge from pretraining to down-
stream few-shot intent detection. QueryAdapt operates at
the relation token level by generating intent-specific relation
tokens from well-learned query-query and query-answer
relations, enabling more fine-grained knowledge transfer.
The main contributions of our paper are summarized as
follows:
```
- Conceptually, we identify two important types of struc-
    tural information inherent in conversational systems and
    examine the knowledge transfer across these relations in
    the context of intent detection, which leads to a new
    structure-aware problem formulation for the few-shot in-
    tent detection problem for the first time.
- Methodologically, we adopt a fundamentally different
    perspective from traditional contrastive learning-based
    methods and propose a novel pure prompt-based ap-
    proach, which features in a flexible relation-aware soft
    prompt module that unifies query text and multiple re-
    lations within a single framework, along with a model-
    agnostic query-adaptive attention network that automati-
    cally generates soft prompts in an adaptive way.
- Empirically, extensive experimental results on two real-
    world datasets across four few-shot settings demon-
    strate that our SAID model significantly and consis-
    tently outperforms state-of-the-art baselines, achieving
    notable improvements of up to 27% in the 3-shot set-
    ting on average. When equipped with the relation to-
    ken–level QueryAdapt module, the enhanced version
    SAID (+QueryAdapt) yields an additional 21% improve-
    ments over SAID in the same setting. Furthermore, we
    validate the flexibility of our framework by showing that
    both SAID and SAID (+QueryAdapt) can serve as plug-
    and-play solutions across various backbone architectures,
    while maintaining strong performance gains.
- Finally, we provide intent annotations for an unlabeled
    conversational dataset derived from the ChatGPT plat-
    form and make it publicly available to facilitate further
    research and development in this field.

## 2 RELATED WORK

```
2.1 Few-shot Intent Detection
Pretraining on large-scale dialogue text corpora is a com-
mon strategy to address the data scarcity problem in few-
shot intent detection task. Earlier efforts in this field are
dedicated to continuously pretraining PLMs by leverag-
ing self-supervised language modeling on large dialogue
datasets and on the domain itself to tackle few-shot intent
detection problem [17], [18]. For example, CONVBERT [18]
constructs an unlabeled dialogue corpus by concatenating
user conversations into long text sequences and continues
pretraining BERT upon this corpus, following standard
```

BERT pretraining settings. Subsequent works extend this
idea by designing additional pretraining tasks, incorporat-
ing either human-labeled intent datasets or pseudo-labeled
query datasets [10], [11], [15], [19]. For example, Intent-
BERT [10] finds that simultaneously optimizing a BERT-
style language modeling loss alongside a supervised loss on
a small human-labeled query intent dataset leads to better
model performance. Shnarch et al. [11] further demonstrate
that unsupervised clustering of query texts can generate
pseudo query labels, which can be used as valuable inter-
mediate signals for model pretraining. Similarly, LLM-Aug
[15] and DFT++ [19] generate pseudo-labeled or unlabeled
augmented user query data by leveraging powerful large
language models such as GPT-3.
Another group of works focuses on adapting contrastive
learning techniques [20], [21] to this domain by designing
various data augmentation strategies for user query data
and pretraining PLMs using contrastive-based objectives.
Specifically, Zhang et al. [22] propose a data augmentation
strategy that pretrains a model on annotated pairs from
natural language inference (NLI) datasets and establishes a
nearest neighbor classification schema for transfer learning,
where two queries from the same intent class are labeled as
positive. In addition to explicitly leveraging human-labeled
data as augmentations, existing methods have also adopted
heuristic and task-adaptive strategies, such as query text
corruption [13] as well as different dropout masks [12].
However, all these methods have primarily focused on
pretraining PLMs using only textual data, failing to ex-
plicitly model the important structural information. Our
work addresses this gap by identifying two crucial types
of structural information and proposing the first structure-
aware pretraining framework in the context of few-shot
intent detection.

**2.2 Prompting Method**

Most widely adopted PLMs, including BERT [1], T5 [23]
and GPT [24], are pretrained using task agnostic language
modeling objectives, which are often inconsistent with the
specific goals of downstream tasks. To this end, prompt-
based learning is proposed to reformulate these tasks as
language modeling problems and facilitate PLMs to adapt
effectively to these downstream tasks [25]. By defining new
prompting functions, prompt-based learning even enables
PLMs to perform few-shot and zero-shot learning, allowing
them to adapt to new scenarios with limited labeled data
[26], [27]. Early methods in this field focus on manually
crafting hard prompts, which are language instructions con-
sisting of discrete text tokens prepended to the input text
[28]. While effective, these hard prompts are challenging to
optimize and rely heavily on domain-specific knowledge.
To address the limitations of manual prompt design, re-
cent works have proposed the use of soft prompts [29],
[30]. Unlike hard prompts, which are composed of static
tokens, soft prompts are continuous, learnable embeddings
that can be optimized automatically via gradients during
training. This approach has proven to be highly effective
in various NLP tasks [31], [32]. Aside from their success in
the language domain, prompting methods have also been
effectively applied in other areas including computer vision

```
[33], [34] and graph neural networks [35], [36]. For instance,
GPPT [36] leverages prompting methods to bridge the gap
between link prediction, which is used as a pretext task
during pretraining, and the downstream node classification
problem.
Our work builds upon these advancements by incorpo-
rating soft prompt learning to enhance the few-shot intent
detection task. By using relation-aware prompts in con-
junction with a structure-aware masked language modeling
task, we aim to improve the model’s ability to capture and
leverage the underlying relational structures embedded in
large-scale dialogue text corpora, thereby facilitating better
generalization and transfer learning in low-data scenarios.
```
## 3 PROBLEM STATEMENT

```
In this section, we formulate the proposed problem of
structure-aware pretraining for few-shot intent detection.
Our problem relies on the existence of a large-scale unla-
beled user dialogue dataset for model pretraining and a
small labeled query intent dataset for model fine-tuning.
Unlabeled Data. Considering a LLM-powered conversa-
tional system where a user uj interacts with the LLM
agent (i.e., chatbot) repeatedly for certain query purpose
at time point t, which constitutes a query session denoted
as Stuj = {(qi,ai)|i = 1,...,N}, where N represents the
length of this session, qirepresents the user query and
ai denotes a unified textual response generated by the
LLM agent. Within each query session Sutj, two important
relations are introduced. Specifically, between two queries
qi and qj within the same session Stuj, the query-query
relation, denoted as rqqis established to reflect the semantic
relevance between them. On the other hand, between a
user query qiand its associated answer ai, the query-answer
relation, denoted as rqa, is also introduced to describe the
correspondence between them. These critical relations are
collectively referred to as structural information within the
conversational systems in this paper. By aggregating query
sessions across different users and time points, we obtain
a large-scale unlabeled user dialogue dataset Dunlabeled=
{Sutj|uj ∈ U,ts≤ t ≤ te}, where U represents sampled
users and [ts,te] represents the observation window.
Labeled Data. We are also given an intent dataset Dlabeled=
{(qi,yi)|i = 1,...,M}, where M represents the number of
labeled queries, yiis the intent label of query qiand it is
one of C pre-defined intent labels denoted as Y = {yi|i =
1 ,...,C}.
In our paper, we focus on a few-shot learning scenario.
Following existing studies [13], we assume that we have K
examples for each of the C classes in our labeled data. In
other words, we have M = C ∗ K training examples in
total.
Problem Definition. Given the unlabeled user dialogue
dataset Dunlabeled and the labeled query intent dataset
Dlabeled, our problem is to learn an intent detection model
fθ(·) that can accurately predict the intent of a new user
query qm.
Note that structural information is only utilized in the
easily collected unlabeled dataset and during the model
pretraining stage, rather than in the fine-tuning or inference
stages. The inference procedure strictly follows the standard
```

few-shot intent detection setup as adopted in prior work
[13], [10], [22], meaning that intent is predicted based on
the given query text, without relying on any historical
information, to ensure both efficiency and fair comparison.
This setup is also particularly important for identifying
user intent at the beginning of a conversation, e.g., the
first query in a new user session. Accurately recognizing
user needs from the outset enables the system to generate
higher-quality responses throughout the session, ultimately
improving overall user satisfaction and engagement.

## 4 METHODOLOGY

In this section, we first present the technical details of SAID,
with its overall architecture illustrated in Figure 1. SAID
comprises two key stages: a pretraining stage, where a
PLM is pretrained to become structure-aware through self-
supervised learning on large-scale unlabeled dialogue text
corpora to enhance query understanding, and a subsequent
supervised fine-tuning stage, where pretrained knowledge
is transferred for few-shot intent detection using prompt-
based task reformulation.
In the pretraining stage, two important structures in-
herent in conversational systems that are often overlooked
by existing works but would significantly enhance query
understanding are identified, namely the query-query relation
and the query-answer relation. These structures are unified
with query text through soft prompt learning, resulting
in a relation-aware prompt design that is easy for PLMs
to interpret and flexible enough to incorporate multiple
relations within a single framework. Building upon this
foundation, a PLM is pretrained using the constructed
relation-aware prompt as input through a structure-aware
masked language modeling task, which serves as a natural
extension of the traditional MLM task within the context of
conversational structures.
In the fine-tuning stage, to facilitate effective knowledge
transfer, we reformulate the intent detection task by intro-
ducing intent-specific relation tokens alongside the semantic
intent names. This creates a new intent-specific relation-
aware prompt, motivated by the idea that the semantics of
different relations can complement each other. The prompt
is then utilized as input for the pretrained PLMs for intent
detection.
Building on this framework, we further propose a novel
mechanism named the query-adaptive attention network
(QueryAdapt) to explicitly transfer relational knowledge
from pretraining to fine-tuning stage by generating intent-
specific relation tokens from well-learned query-query and
query-answer relations. This approach leverages the seman-
tic proximity of relational structures to further enhance
model performance under limited supervision.

**4.1 Structure-aware Pretraining with Prompt Learning**

Due to the inherently goal-driven and often informal or
noisy nature of user dialogues [37], [38], models pretrained
on other text corpora (e.g., BERT) struggle to effectively
handle user queries in conversational systems. To address
the semantic gap, these PLMs are continuously pretrained
using dialogue text corpora, with a primary focus on the

```
textual components, to enable them adapt to the unique
characteristics of user queries [18], [39], [13], [10]. However,
apart from the essential textual information, the multi-turn
structure is also a unique characteristic of queries in con-
versational systems, as users commonly refine their queries
through iterative exchanges on these platforms. Such multi-
turn structure can be understood from two fundamental re-
lational perspectives: the query-query relation and the query-
answer relation as introduced in Section 3.
```
```
4.1.1 Relation Overview
On the one hand, the query-query relation, denoted as rqq,
occurs between two queries posted by the user within the
same session, reflecting how the user refines her requests
and helping to interpret the user query from a human refine-
ment perspective. On the other hand, the query-answer rela-
tion, denoted as rqa, describes the correspondence between a
user query and its answer, illustrating how the model (often
an LLM in modern conversational systems) responds to the
user’s query and helping to interpret the user query from
a model clarification perspective. They collectively enhance
the understanding of a given query from various relational
perspectives, providing valuable insights for interpreting a
user query from the intent perspective, as the associated
knowledge among them is highly relevant.
```
```
4.1.2 Relation-aware Prompt Learning
To enable PLMs to be explicitly aware of these structures
beneficial for query understanding while maintaining their
ability to address the semantic gap between user query text
and other common text through pretraining on dialogue
text corpora, we propose to unify these two aspects (i.e.,
relations and text) together for model pretraining and adopt
novel soft prompt learning to implement this idea. This
represents the first attempt in this field to the best of our
knowledge. Specifically, we introduce relation tokens which
function as specialized soft prompts and propose a relation-
aware prompt that is easy for PLMs to interpret and helps
them grasp knowledge about the unique textual character-
istics of queries and the multi-turn structures inherent in
conversational systems simultaneously.
Given a multi-turn query session Stujinitiated by user uj
at time point t, we construct two relation-aware prompts for
each query qi∈ Stuj, corresponding to the two fundamental
relations identified above as follows:
Promptqq={qi;z(1)qq,··· ,z(qqm); qj},
Promptqa={qi;z(1)qa,··· ,z(qam); ai},
```
### (1)

```
where {·;·} is token concatenation operation,
{z(1)qq,··· ,z(qqm)} represent the m relation tokens introduced
for the query-query relation rqq, and eachz(qqm) is a
k-dimensional trainable embedding vector; similarly,
{z(1)qa,··· ,z(qam)} denote the corresponding relation tokens
for the query-answer relation rqa. These relation tokens
are designed to act as specialized soft prompts in our
framework. qj ∈ Sutj is another user query in the same
session as qi, and airefers to the response generated by the
platform for query qi. This prompt design integrates both
relations and text that are essential for query understanding
into a single framework. Note that our method is flexible
```

```
New Query Session
```
```
Candidate Intent Name 𝑦𝑘
```
```
Are there any long-term effects on marine life
due to the melting ice and climate change?
```
```
Text Creation
```
```
Query-Intent Relation 𝑟𝑞𝑖
```
```
Can you explain how climate change affects
polar ice caps?
Climate change causes global temperatures
to rise, which leads to the melting of polar
ice caps.
Will climate change contribute to global sea
level rise?
```
```
Yes. The melting of ice caps adds large volumes
of freshwater to the oceans, increasing global
```
## sea levels. ...

```
Are there any long-term effects on marine life
due to the melting ice and climate change?
```
```
Query-Answer Relation 𝑟𝑞𝑎
```
```
Query-Query Relation 𝑟𝑞𝑞
```
```
......
```
```
Text Tokens for 𝑞𝑖Relation Tokens for 𝑟𝑞𝑞 Text Tokens for 𝑞𝑗
```
```
...... ......
```
```
...... ......
```
```
Text Tokens for 𝑞𝑖Relation Tokens for 𝑟𝑞𝑎Text Tokens for 𝑎𝑖
```
```
......
```
```
Multiple Relation-awareSoft Prompts
(a) Structure-aware Pretraining with Prompt Learning
```
```
......
```
```
Text Tokens for 𝑞𝑚Relation Tokens for 𝑟𝑞𝑖Text Tokens for 𝑦𝑘
```
```
...... ......
```
```
BackboneLanguage Models
```
```
Structure-awareMLM
```
```
Structure-aware Pretrained
Language Models
```
```
(b) Prompt-based Fine-tuning for Few-shot Intent Detection
```
```
ProbabilityDistribution logit𝑘
```
```
Task Reformulation
```
```
Intent-specific Relation-awarePrompt:
```
```
Structural Information
```
```
𝑞𝑗
```
```
𝑞𝑖
```
```
𝑎𝑖
```
```
Query 𝑞𝑚 in a
New Session
```
Fig. 1: The model architecture of SAID. (a) Pretraining stage with relation-aware soft prompts; (b) Fine-tuning and inference
stage with prompt-based task reformulation.

and can be easily extended to accommodate more relations,
such as similar queries explicitly labeled by humans.

4.1.3 Structure-aware Masked Language Modeling

For each query in the unlabeled dialogue text corpora, we
generate two corresponding relation-aware prompts, result-
ing in our structure-aware pretraining dataset. A PLM is
then pretrained on this mixed dataset using the masked lan-
guage modeling loss to grasp knowledge about the textual
characteristics of queries and the multi-turn structures as
follows:

```
L =−
```
### 1

### M

### XM

```
i=
```
```
logP(xi|X\i), (2)
```
where M is the number of masked tokens, xiis the text
token at position i, X\irepresents the input sequence with
the i-th text token masked. This pretraining step serves
as a natural extension of the traditional MLM task in
BERT, which relies solely on text data, by incorporating the
proposed relation-aware prompts as inputs. It enables the
model to learn how to interpret query text within the context
of various relations.

**4.2 Few-shot Intent Detection**

4.2.1 Implicit Knowledge Transfer via Task Reformulation

Through self-supervised learning in the first stage, the
model efficiently leverages a large number of unlabeled user
queries alongside conversational structures. In the second
stage, the model undergoes further tuning in a supervised

```
manner using limited examples for few-shot intent detec-
tion. The fine-tuning design of our model is guided by the
intuition that query intent detection can be framed as a
query understanding task from the perspective of intent re-
lation, which closely relates to both query-query and query-
answer relations already exposed to and well captured
by the PLMs during the pretraining phase. Therefore, to
facilitate effective knowledge transfer, we propose to first
reformulate the intent detection task from the relational
perspective.
Specifically, given a new user query qm, we introduce
the intent name as semantic information and build a new
relation-aware prompt for each intent class as follows:
Promptqi={qm;z(1)qi,··· ,z(qim); yk}, where k = 1,...,C.
(3)
Here, {z(1)qi,··· ,z(qim)} are new relation tokens introduced
specifically for the query-intent relation (i.e., the relation rqi
between qmand ykshown in Figure 1), ykis the intent name
for k-th intent class.
Instead of relying solely on user query text (i.e., qm) as
done in existing works, we utilize the intent-specific prompt
as input to the pretrained PLMs for classification. The prob-
ability distribution of classifying qminto each intent class is
calculated as follows:
logitmk = MLP
```
### 

```
fplm
```
### 

```
Promptqi
```
### 

### ,

```
pm= Softmax
```
### 

```
logitm 1 ,··· , logitmC
```
### 

### ,

### (4)

```
where fplm(·) denotes the pretrained PLM. This model is
then trained on the few-shot examples using cross-entropy
```

loss. During inference, we predict the intent with the highest
probability as the final output. We refer to this model as
**SAID** , which also serves as our default model.

### ......

```
Text Tokens for 𝑞#Relation Tokens for 𝑟!" Text Tokens for 𝑦$
```
### ...... ......

```
Structure-aware Pretrained
Language Models
```
```
ProbabilityDistribu-on logit$ Default Model: SAID
```
### ......

```
Relation Tokens for 𝑟!!
```
### ......

```
Relation Tokens for 𝑟!%
```
```
PLM
Encoder 𝝀𝒒𝒒 𝝀𝒒𝒂
```
### ......

```
Pretrained Relation Token Pool
```
```
Soft Prompt Generation
Query-Adaptive
Attention Network
```
```
Fig. 2: The model architecture of SAID (+QueryAdapt).
```
4.2.2 Explicit Knowledge Transfer via Soft Prompt Genera-
tion

By reformulating query intent detection from a relational
perspective, we improve its alignment with the pretrain-
ing task, thereby facilitating the knowledge transfer of the
overall modeling framework at the modeling paradigm
level. However, the newly introduced intent-specific relation
tokens are randomly initialized and may not be effectively
optimized during the fine-tuning process with limited few-
shot examples. To further enhance the knowledge transfer
from the pretraining stage to the fine-tuning stage, we
propose to transfer relation tokens explicitly at the fine-
grained relation token level. The core intuition is that the
query-intent relation is semantically closely related to the
pretrained query-query and query-answer relations, allow-
ing it to borrow knowledge from these well-learned rela-
tions. Specifically, we enable intent-specific relation tokens
to be adaptively generated from the query-query and query-
answer relation tokens, which have been effectively learned
during the model pretraining phase.
Formally, given a new user query qm, the corresponding
query-intent relation tokens {z
(1)
qi,··· ,z

(m)
qi } are generated
by aggregating information from the pretrained query-
query relation tokens {z(1)qq,··· ,z(qqm)} and query-answer
relation tokens {z(1)qa,··· ,z(qam)} using a query-adaptive at-
tention network as follows:

```
z(qim)= λqq·z(qqm)+ λqa·z(qam),
[λqq,λqa] = Softmax (MLP (fplm(qm))),
```
### (5)

where λqqand λqaare the weights for query-query and
query-answer relations respectively, generated based on
the given user query qmencoded by the pretrained PLM

```
TABLE 1: Representative examples in IntentChat and Wild-
Chat.
```
```
Query Intent Dataset Examples
Text Creation IntentChatWildChat Write a letter of complaint to the local council. 140 words using the写老人步履蹒跚的句子。
following prompts: recently you parked car/ returned/parking fine/noparking sign behind tree.
Knowledge Q & A IntentChatWildChat While bioelectrochemistry deals with the electron transfer processes红薯煮几分钟。
of biological systems, what types of biological systems are specificallystudied and why?
Text Understanding IntentChatWildChat Summarize ”Toto is a leading Japanese manufacturer of bathroom你能分析一下这诗句的意境吗
ceramic ware, with annual worldwide sales of around $5 bn. One ofits best-selling ranges is the Washlet lavatory, priced at up to $5,000 and
used in most Japanese homes...
Mathematical Reasoning IntentChatWildChat An engineering system has two independent processes that occur one部分相干月牙形光束在海洋湍流中的交叉谱密度计算公式。
right after the other. BOTH must be successful for the system to work.Both systems work 90% of the time. What is the probability that the
entire system will succeed?
Code IntentChatWildChat how to get the length of a lol, lol2 list in pythonWINCC.7.0如何批量替换。
Casual Conversation IntentChatWildChat why are you not responding. reply shortly.我在人海遇见你，却又转身不见你。
Multimodal IntentChatWildChat —As a prompt generator for a generative AI called ”Midjourney”, you will
create image prompts for the AI to visualize. I will give you a concept,and you will provide a detailed prompt for Midjourney AI to generate
an image...
Ambiguous Requests IntentChatWildChat —you gpt
```
```
fplm(·). The intuition behind our model design is that
different user queries may rely on these relations to varying
extents when inferring user intent. For instance, longer and
noisier queries might benefit more from query-query rela-
tions, leveraging query refinement to more clearly elucidate
their underlying intents.
Based on Equation (5), we construct the intent-specific
relation-aware prompt Promptqias introduced in Equation
(3) using the generated relation tokens, and ultimately pre-
dict the intent of a given user query qmusing Equation (4).
We name this advanced model as SAID (+QueryAdapt) ,
and an overview of its framework is provided in Fig. 2
(with the newly added soft prompt generation components
highlighted in red).
```
```
TABLE 2: Prompt Description
```
```
You are a diligent and responsible dataset annotator. Your task is to read the questions or requests input by users andselect the most relevant label from eight new categories. The eight new categories are as follows:
```
1. **Text Creation**, such as creative writing, informative writing, evaluative writing, and editing, shortening, orexpanding a text. The difference from text understanding is that text creation focuses on generating new content.
2. **Knowledge Q & A**, such as explanatory questions, objective knowledge questions, opinion questions, and termsearches.
3. **Text Understanding**, such as extracting information from a given text, semantic understanding, structural analysis,or language translation. Summarization belongs to text understanding, while shortening belongs to text creation. The
distinction between summarization and shortening is that shortening needs to retain literary qualities.4. **Mathematical Reasoning**, such as performing mathematical calculations, unit conversions, logical reasoning, or
building models.5. **Code**, such as programming problem-solving, code debugging, algorithm design, or data analysis.
6. **Casual Conversation**, such as everyday conversations, personal feelings and evaluations, role-playing, or sharing
life experiences.7. **Multimodal**, such as generating or understanding images, videos, audio, or describing image elements for text-to-
image generation.8. **Ambiguous Requests**, such as requests where the intention is unclear, hard-to-classify intentions, or long texts
without a clear request.The user input is: inputtext, and your output format should be in the JSON format:
    { ”summary”: [summary is less than 80 words],
       ”label”: [label]
Label the intent category for the following data based on the above requirements.}
{inputtext}

## 5 EXPERIMENTS

```
In this section, we conduct extensive experiments to an-
swer the following important research questions regarding
our default model SAID, and its enhanced variant SAID
(+QueryAdapt). Specifically, for the default model SAID:
```
- **RQ1:** Can SAID achieve superior performance compared
    to state-of-the-art methods in intent detection accuracy
    across various few-shot settings?
- **RQ2:** Can SAID be applied as a plug-and-play framework
    to different PLM-based backbones, and how does it per-
    form with these extensions?


TABLE 3: Results for intent detection across different few-shot settings. Tabular results are in percent. For each dataset, the
first row reports the performance of strong zero-shot LLM baselines (such as LLaMA, Qwen, and Mistral) for reference.
The best result is shown in boldface, and the second best result (excluding these LLM models) is underlined.

```
Model
```
```
3-shot 5-shot 10-shot 20-shot
Accuracy F1 Accuracy F1 Accuracy F1 Accuracy F
Dataset: IntentChat
Llama-3.1-8B (zero-shot) - - - - - - 25.08 29.
Qwen-7B (zero-shot) - - - - - - 72.57 75.
BERT 34.98 41.34 33.06 39.76 43.29 46.85 57.53 61.
RoBERTa 40.59 47.46 43.60 48.19 51.57 55.99 62.78 67.
DistilBERT 31.10 36.63 43.24 47.43 52.01 56.96 56.24 60.
ALBERT 32.22 37.87 35.13 39.13 42.36 48.30 44.17 50.
CPFT 39.77 40.04 41.92 40.35 35.00 38.81 43.39 43.
IntentBERT 37.60 42.19 49.16 55.03 59.39 64.34 66.50 70.
DNNC 29.99 43.49 40.86 44.66 49.21 55.35 58.31 64.
SetFit 40.04 40.78 45.45 47.58 54.24 57.74 60.69 64.
LLM-Aug 29.58 35.77 39.60 44.31 43.25 46.58 59.55 63.
DFT++ 20.09 16.24 22.19 16.99 29.27 23.64 38.60 29.
SAID 52.73 58.49 63.75 67.80 69.18 72.54 74.05 76.
Improvement 29.91% 23.24% 29.68% 23.21% 16.48% 12.74% 11.35% 7.92%
Dataset: WildChat
Llama-3.1-8B (zero-shot) - - - - - - 52.01 53.
Mistral-7B (zero-shot) - - - - - - 50.94 50.
BERT 22.88 21.96 35.44 36.95 40.13 43.00 49.16 53.
RoBERTa 33.87 35.61 41.85 45.05 50.15 54.29 53.48 57.
DistilBERT 28.35 30.32 34.30 36.28 38.96 41.93 45.93 50.
ALBERT 30.87 34.35 35.11 39.20 45.45 50.46 47.73 51.
CPFT 31.86 26.95 25.88 25.72 33.84 27.34 34.39 28.
IntentBERT 30.14 34.16 44.41 49.47 47.72 51.93 53.99 59.
DNNC 33.26 34.71 33.18 36.26 49.46 53.03 55.98 59.
SetFit 30.82 33.71 38.73 42.24 46.28 51.13 52.34 57.
LLM-Aug 39.36 42.78 42.18 45.60 45.13 47.75 50.76 53.
DFT++ 40.23 30.60 41.19 35.80 42.85 36.95 49.63 40.
SAID 50.03 53.05 51.20 54.73 57.17 63.01 56.95 62.
Improvement 24.36% 24.01% 15.29% 10.63% 14.00% 16.06% 1.73% 4.08%
```
- **RQ3:** How do different modules of SAID contribute to
    enhancing the overall model performance?
- **RQ4:** How do different hyper-parameter settings impact
    the performance of SAID?
- **RQ5:** How efficient is SAID compared to state-of-the-art
    few-shot intent detection methods?
For the enhanced variant, SAID (+QueryAdapt):
- **RQ6:** Does SAID (+QueryAdapt) outperform the default
    SAID model across various few-shot settings?
- **RQ7:** Can the proposed explicit knowledge transfer mod-
    ule via soft prompt generation (i.e., QueryAdapt) be ap-
    plied as a general strategy across different PLM-based
    backbones to further enhance model performance over the
    default SAID framework?
- **RQ8:** What do the learned attention weights represent?
    Do they effectively capture the underlying characteristics
    of the query intent detection problem?

**5.1 Experimental Settings**

5.1.1 Datasets

Most publicly available datasets are typically limited to
independent query data, lacking the structural information

```
associated with query sessions. By collaborating with a
LLM-powered technology company, we acquire user query
data from its chatbot platform, covering the period from Jan-
uary 2024 to June 2024, resulting in the first dataset, named
IntentChat. Additionally, we utilize a publicly available
unlabeled user conversation dataset from a recent paper
[40], which is the most relevant publicly available dataset
in this domain to the best of our knowledge, though it only
contains conversations from ChatGPT platform without any
user intent annotations. We annotate this dataset with intent
labels, creating the second dataset, named WildChat. This
labeled dataset is made publicly available for the research
community. The representative examples of each query in-
tent are listed in Table 1.
```
```
IntentChat Dataset. During the pretraining stage, we pre-
process the sampled data by removing queries with zero
length and filtering out sessions with fewer than three user
interactions. This results in an unlabeled dataset consisting
of 7,817 user query sessions, which includes a total of 91,
query-answer pairs. We are also provided with an internally
human-annotated dataset, from which we sample 20,
queries to create our evaluation dataset. This dataset is
```

TABLE 4: Results for plug-and-play experiments using three different PLM-based backbones. Tabular results are in percent.

```
Model
```
```
3-shot 5-shot 10-shot 20-shot
Accuracy F1 Accuracy F1 Accuracy F1 Accuracy F
Dataset: IntentChat
BERT 34.98 41.34 33.06 39.76 43.29 46.85 57.53 61.
SAID (BERT) 52.73 58.49 63.75 67.80 69.18 72.54 74.05 76.
Improvement 50.75% 41.48% 92.85% 70.53% 59.81% 54.81% 28.71% 24.15%
DistilBERT 31.10 36.63 43.24 47.43 52.01 56.96 56.24 60.
SAID (DistilBERT) 50.17 53.41 56.24 60.38 59.66 63.87 65.94 69.
Improvement 61.35% 45.82% 30.08% 27.29% 14.72% 12.13% 17.24% 13.63%
ALBERT 32.22 37.87 35.13 39.13 42.36 48.30 44.17 50.
SAID (ALBERT) 49.51 50.36 50.62 54.83 61.27 64.22 57.32 61.
Improvement 53.67% 32.99% 44.08% 40.13% 44.66% 32.95% 29.78% 21.84%
Dataset: WildChat
BERT 22.88 21.96 35.44 36.95 40.13 43.00 49.16 53.
SAID (BERT) 50.03 53.05 51.20 54.73 57.17 63.01 56.95 62.
Improvement 118.68% 141.61% 44.47% 48.14% 42.45% 46.52% 15.85% 16.21%
DistilBERT 28.35 30.32 34.30 36.28 38.96 41.93 45.93 50.
SAID (DistilBERT) 34.38 33.29 45.90 50.14 50.43 54.88 49.48 53.
Improvement 21.27% 9.79% 33.83% 38.21% 29.44% 30.88% 7.74% 6.59%
ALBERT 30.87 34.35 35.11 39.20 45.45 50.46 47.73 51.
SAID (ALBERT) 43.30 48.47 52.03 56.97 58.75 62.89 59.03 64.
Improvement 40.25% 41.08% 48.20% 45.35% 29.26% 24.64% 23.68% 24.81%
```
used during the fine-tuning and testing stages. Following
the experimental setup outlined by existing works [13], we
sample K -shot labeled queries per intent from the human-
annotated dataset to fine-tune our model, while the remain-
ing queries are regarded as the validation and test set. Note
that we utilize user data with full consent obtained and in
accordance with relevant privacy and ethical guidelines.
**WildChat Dataset.** For the pretraining dataset, we first re-
tain only the English conversations, and then apply the same
preprocessing steps as introduced in IntentChat, resulting
in an unlabeled dataset consisting of 62,949 user query
sessions, which includes a total of 354,104 query-answer
pairs. For the evaluation dataset, we hold out and annotate
10,000 queries using GPT-4 Turbo. The prompt used for this
labeling process is provided in Table 2. To ensure annotation
quality, we engaged two student experts to manually review
the labeled queries, correct any identified errors, and ensure
consistency in their opinions.

5.1.2 Evaluation Metrics

Accuracy and F1 are used to evaluate the effectiveness
of the few-shot intent classification task following existing
works [41]. All experiments are repeated five times, and
the average results are reported to ensure robustness of the
evaluation.

5.1.3 Baselines

We compare our approach against **thirteen** strong baseline
models reported in the literature. Note that all methods
adopt the same inference procedure and predict the intent
of a given user query based on the input query text, thereby
making it an inherently fair comparison.

```
The first category follows a classifier architecture that
learns query representations using various PLM backbones,
followed by a linear classification head optimized through
cross-entropy loss. We include four commonly adopted
PLMs in our experiments: BERT [1], RoBERTa [14], Dis-
tilBERT [42] and ALBERT [43]. Specifically, BERT is a
foundational transformer-based model that captures deep
bidirectional representations, serving as the basis for numer-
ous downstream NLP tasks. RoBERTa builds upon BERT by
improving the pretraining approach through larger datasets
and a dynamic masking strategy. DistilBERT , on the other
hand, distills the knowledge of BERT into a more compact
model, providing a balance between speed and accuracy
while retaining the essential capabilities of the original
BERT architecture. ALBERT is another widely adopted
lightweight variant of BERT that reduces model size through
parameter sharing and factorized embedding, achieving
competitive performance with improved efficiency.
The second category comprises methods specifically de-
signed for the few-shot intent detection problem, including
CPFT [13], IntentBERT [10], DNNC [22] SetFit [44], LLM-
Aug [15] and DFT++ [19]. Specifically, CPFT is a two-stage
pretraining framework for intent detection. In the first stage,
the model learns using a self-supervised contrastive loss on
a large set of unlabeled queries with data augmentations.
In the second stage, it uses a supervised contrastive loss
to consolidate query representations from the same intent
class. IntentBERT pretrains BERT via supervised learning
on a small human-labeled query intent dataset. DNNC
is trained to identify the best-matched example from the
training set through similarity matching. It enhances per-
formance by incorporating data augmentations and natural
language inference tasks during pretraining. SetFit is a
```

Sentence Transformer model specifically designed for few-
shot classification tasks. It uses triplet loss to fine-tune
the transformer encoder and cross-entropy loss to optimize
the classification head sequentially. LLM-Aug employs data
augmentation to generate labeled data for intent detec-
tion by prompting large language models. Finally, DFT++
introduces a context augmentation method and leverages
a sequential self-distillation technique to improve model
performance.
Finally, the third category includes zero-shot large lan-
guage model (LLM) baselines using the prompts shown in
Table 2. For the IntentChat dataset, we include **LLaMA-3.1-
8B** and **Qwen-7B** ; for the WildChat dataset, we evaluate
**LLaMA-3.1-8B** and **Mistral-7B**. Among them, LLaMA-3.1-
8B is one of the most widely adopted general-purpose
open-source LLMs. Qwen-7B is specifically optimized for
Chinese understanding and generation, while Mistral-7B
demonstrates strong performance on a wide range of En-
glish benchmarks. These models are approximately 60–
times larger than BERT, our default backbone model.

5.1.4 Reproducibility

Our method is designed as a flexible, plug-and-play frame-
work that can be seamlessly integrated with any PLM
backbone. To balance effectiveness and efficiency, and in line
with previous works [13], [10] to ensure a fair comparison,
we adopt the widely used BERT-base model as the default
backbone for our experiments. In the pretraining stage, we
pretrain our model for 4 epochs using the Adam optimizer
with a learning rate of 3e-5. For the MLM loss, we follow
the masking strategy used in BERT [1]. The number of
relation tokens is set to 3. In the fine-tuning stage, we con-
duct experiments across four few-shot learning scenarios,
specifically 3-shot (3 training examples per intent), 5-shot,
10-shot, and 20-shot settings. A hyperparameter search is
performed for the learning rate over{1e-5, 4e-5, 1e-4}, using
an early stopping strategy to prevent overfitting. Likewise,
for the baseline methods, hyperparameters are carefully
tuned according to their best performances on the validation
set. All experiments are conducted on a single NVIDIA A
GPU with 80GB of memory.

**5.2 Performance Comparison (RQ1)**

The main results on two datasets are provided in Table 3,
where the best performance is in boldface and the second
best (excluding LLM models such as LLaMA, Qwen, and
Mistral) is underlined, and the improvement of SAID over
these second best methods is also presented in the last row.
The zero-shot performance of the LLM models is included
as a reference in the first row.
From these results, we can make the following observa-
tions. (1) The performance of RoBERTa is generally superior
to that of BERT, which can be attributed to the fact that
RoBERTa is trained with more robust optimizations and on
larger datasets, providing it with stronger modeling capa-
bilities compared to BERT. Similar findings can be observed
by comparing BERT with DistilBERT where BERT has more
parameters and exhibits more powerful capabilities. (2)
The few-shot intent detection models generally achieve the
best performances in most cases among all baselines. For

```
instance, even with BERT as the backbone, IntentBERT,
pretrained on large-scale dialogue text corpora, effectively
adapts to the unique characteristics of user queries and
bridges the semantic gap, achieving even better perfor-
mance than RoBERTa. (3) Unlike these baselines that focus
solely on text, we introduce additional structural informa-
tion beneficial for query understanding during pretraining,
enabling us to fully leverage the unlabeled dialogue text
corpora. Meanwhile, we reformulate the few-shot intent de-
tection problem from a relational perspective using prompt
learning during fine-tuning, facilitating effective knowledge
transfer from pretrained relations. As demonstrated in Table
3, these novel designs enable our model to consistently out-
perform all the baselines across all metrics on both datasets.
And the average margin in terms of accuracy (F1 score)
reaches 21.86% (16.78%) for the IntentChat dataset and
13.85% (13.70%) for the WildChat dataset, respectively. (4)
Even with a small PLM backbone (BERT), our model SAID
achieves comparable or even superior performance to LLMs
that are 60–70 times larger, using very limited supervision
(e.g., 5-shot labeled data on the WildChat dataset). This re-
sult underscores the effectiveness of our proposed relation-
aware prompt-based pretraining and finetuning framework.
Furthermore, equipped with an explicit knowledge transfer
strategy based on soft prompt generation, our model can
achieve better performance compared to these LLMs, while
requiring fewer labeled examples. Additional results are
provided in Section 5.7. Given the substantial computa-
tional and financial costs associated with these LLMs, our
approach offers a highly cost-effective, scalable, and better
alternative for real-world deployment.
```
```
5.3 Extension to Various Backbones (RQ2)
To verify the plug-and-play capability and robustness of
our framework, we replace the backbone model (BERT-
base) with other commonly adopted PLMs, including both
DistilBERT and ALBERT. The experiment results on both
datasets are presented in Table 4, where the performance im-
provement of the SAID framework over each corresponding
backbone model is presented. Despite variations in archi-
tecture and/or model size between BERT, DistilBERT, and
ALBERT, our framework consistently delivers significant
improvements across all PLMs in all few-shot settings. We
also observe that, when equipped with SAID, previously
underperforming models such as DistilBERT and ALBERT
could outperform the best baseline in Table 3. For instance,
SAID (DistilBERT) and SAID (ALBERT) achieve much better
performance in the 5-shot setting on the WildChat dataset
compared to IntentBERT (the strongest baseline exclud-
ing LLM models), demonstrating that SAID can effectively
adapt to different PLMs while maintaining strong perfor-
mance. Overall, these results validate the robustness and
plug-and-play capability of our proposed framework.
```
```
5.4 Ablation Study (RQ3)
To demonstrate how different components of the proposed
framework affect the performance, we conduct ablation
studies by removing or replacing key modeling modules
from the full architecture. Specifically, we investigate five
simplified versions of our model: (1) w/o PT removes the
```

```
TABLE 5: Results for ablation study on IntentChat. Tabular results are in percent.
```
## Model

## 3-shot 5-shot 10-shot 20-shot

## Accuracy F1 Accuracy F1 Accuracy F1 Accuracy F

## w/o PT 45.92 52.52 57.18 61.62 55.28 59.15 65.88 69.

## w/o REL 46.51 52.09 56.54 61.80 59.82 64.46 67.30 71.

## w/o Q-A 51.68 55.73 58.85 63.50 62.15 67.51 64.91 70.

## w/o Q-Q 54.24 60.50 57.75 63.47 66.63 71.03 69.72 73.

## w/o PLFT 41.26 46.85 43.72 49.15 55.37 60.06 64.48 68.

## SAID 52.73 58.49 63.75 67.80 69.18 72.54 74.05 76.

entire pretraining phase; (2) **w/o REL** excludes all structural
information during pretraining, relying solely on unlabeled
query text to pretrain BERT; (3) **w/o Q-A** and **w/o Q-Q**
eliminate the query-answer relation and the query-query
relation during pretraining, respectively; (4) Finally, **w/o
PLFT** replaces the proposed prompt-learning based fine-
tuning approach with a traditional classification head that
relies solely on query text for intent classification (as done
in existing works). The ablation results on the IntentChat
dataset are presented in Table 5.
**Benefits of structure-aware pretraining.** From the results
summarized in Table 5, we can make the following observa-
tions. (1) In most cases, the performance of w/o REL exceeds
that of w/o PT, revealing that pretraining with unlabeled
query text helps bridge the semantic gap and enables the
PLM to adapt to the unique characteristics of user queries
in conversational systems. (2) However, w/o REL is often
outperformed by w/o Q-A and w/o Q-Q. For example,
w/o Q-Q surpasses w/o REL by approximately 11.38%
and 10.19% in terms of average accuracy and F1 score
respectively in the 10-shot setting, showing that incorpo-
rating additional structural information during pretraining
is crucial for better query understanding. (3) In general, the
full model SAID significantly outperforms both w/o Q-A
and w/o Q-Q, demonstrating the advantages of integrating
various critical relations in conversational systems within a
unified framework. Overall, these experiments collectively
highlight the importance of our proposed structure-aware
pretraining framework.
**Benefits of prompt-learning based fine-tuning.** Compared
to w/o PLFT, SAID achieves better performance in all
metrics and across all few-shot settings, with an average
improvement of 28.35% in accuracy and 23.64% in F1 score,
respectively. These results verify the necessity of prompt-
learning based task reformulation for effective knowledge
transfer and highlight its advantages over traditional solu-
tions.

**5.5 Parameter Analysis (RQ4)**

In this section, we investigate the impacts of two important
hyperparameters, the masking ratio and the number of
pretraining epochs, on SAID using the IntentChat dataset.
**Impact of Masking Ratio.** As shown in Figure 3, the
model performance initially declines as the masking ratio
increases, but starts to improve after it exceeds 0.20. The best
performance across most metrics is achieved at a masking

```
(a) Accuracy (b) F
Fig. 3: Impact of masking ratio on IntentChat.
```
```
(a) Accuracy (b) F
Fig. 4: Impact of pretraining epochs on IntentChat.
```
```
ratio of mr = 0. 25 , which is different from the optimal value
of mr = 0. 15 found in BERT’s standard masked language
modeling task. This suggests that by introducing additional
relation contexts, a more challenging pretraining setting is
needed for improved representation learning.
Impact of Pretraining Epochs. Figure 4 illustrates the im-
pact of pretraining epochs on the model performance. It’s
evident that, in most cases, model performance improves as
the number of pretraining epochs increases. For example,
the average accuracy in the 20-shot setting when pe = 4
is over 10% higher than when pe = 1, which validates
the benefits of our structure-aware pretraining from another
perspective.
```
```
5.6 Efficiency Analysis (RQ5)
In this section, we compare the training and inference time
of the proposed model against several baseline methods
designed for few-shot intent detection, with a focus on
efficiency analysis. Following existing studies, we evaluate
and report the average training time per epoch [45] and
the average inference time per query (commonly referred
to as inference latency). The experimental results using the
IntentChat dataset are summarized in Table 6.
Training efficiency. The proposed models, SAID (BERT)
and SAID (ALBERT), demonstrate competitive training ef-
ficiency compared to the baselines. With training times
```

```
TABLE 6: Results for efficiency analysis on IntentChat.
```
```
Model
```
```
Average Training Time per Epoch (s) Average Inference Time per Query (s)
3-shot 5-shot 10-shot 20-shot N/A
CPFT 2.61 2.71 2.87 3.24 2.6e-
IntentBERT 0.19 0.20 0.32 0.77 3.5e-
DNNC 5.59 12.09 40.78 149.53 0.
SetFit 6.44 7.68 11.81 20.42 7.7e-
LLM-Aug 0.25 0.43 0.80 1.59 4.6e-
DFT++ 27.82 29.92 32.88 39.35 0.
SAID (BERT) 2.25 3.51 7.14 15.38 3.3e-
SAID (ALBERT) 1.71 2.51 5.19 10.32 2.4e-
```
TABLE 7: Results of soft prompt generation for intent detection across various few-shot settings. Tabular results are in
percent.

```
Model
```
```
3-shot 5-shot 10-shot 20-shot
Accuracy F1 Accuracy F1 Accuracy F1 Accuracy F
```
```
Dataset: IntentChat
SAID 52.73 58.49 63.75 67.80 69.18 72.54 74.05 76.
+ Linear 65.86 68.84 71.58 74.14 70.33 74.09 69.25 73.
+ MLP 61.75 67.17 68.28 72.60 70.51 74.67 71.24 75.
+ QueryAdapt 67.62 69.89 73.07 75.92 75.03 77.88 72.08 75.
Improvement 28.24% 19.49% 14.62% 11.98% 8.46% 7.36% -2.66% -0.89%
```
```
Dataset: WildChat
SAID 50.03 53.05 51.20 54.73 57.17 63.01 56.95 62.
+ Linear 53.65 58.40 52.54 57.58 58.04 62.17 60.11 64.
+ MLP 45.19 48.79 56.58 60.11 54.62 58.91 56.53 61.
+ QueryAdapt 56.58 61.64 58.08 63.75 60.11 64.01 65.18 68.
Improvement 13.09% 16.19% 13.44% 16.48% 5.14% 1.59% 14.45% 9.45%
```
ranging from 1 to 15 seconds per epoch, these models are
computationally efficient and well within acceptable limits,
particularly given that training is conducted offline.
**Inference efficiency.** In terms of the more important in-
ference latency, both SAID (BERT) and SAID (ALBERT)
demonstrate comparable or even superior efficiency to the
most efficient baseline CPFT, while providing significant
improvements in terms of intent detection performance.
This characteristic makes them ideal for online services
that often require low-latency, real-time, and high-quality
responses.

**5.7 Effectiveness of SAID (+QueryAdapt) (RQ6)**

In this section, we investigate the explicit knowledge trans-
fer strategy based on soft prompt generation. Building upon
the SAID framework through task reformulation, we further
propose to explicitly transfer relation tokens by generating
intent-specific relation tokens explicitly during model fine-
tuning and inference. In addition to the query-adaptive
attention network (i.e., QueryAdapt), we also design two ad-
ditional simple transfer strategies, namely Linear and MLP
in this section. Specifically, the Linear strategy introduces
two global weights, λqqand λqadefined in Equation (5),

```
which are shared across all user queries and optimized
during model fine-tuning. On the other hand, the MLP
strategy utilizes the query-query and query-answer relation
tokens as input and generates intent-specific relation tokens
through a feedforward deep neural network. The experi-
mental results on the two datasets are summarized in Table
7.
Effectiveness of the explicit knowledge transfer strategy.
In general, the proposed explicit knowledge transfer strat-
egy via soft prompt generation can help to improve the over-
all model performance. As shown in Table 7, in most cases,
the Linear, MLP, and QueryAdapt extensions significantly
outperform their counterpart, the default model SAID. For
example, SAID (+Linear) outperforms SAID by 24.90% in
terms of Accuracy and 17.70% in terms of F1 score under the
3-shot setting on the IntentChat dataset. Similar findings can
also be observed when comparing SAID (+MLP) and SAID
(+QueryAdapt) with the default model SAID.
Effectiveness of the query-adaptive attention network.
Compared with the simple transfer strategies (i.e., Linear
and MLP), our proposed query-adaptive attention network
achieves the best performance on two datasets across all
the studied settings. The main difference between the Lin-
```

```
TABLE 8: Results for the QueryAdapt strategy on three different PLM-based backbones. Tabular results are in percent.
```
```
Model
```
```
3-shot 5-shot 10-shot 20-shot
Accuracy F1 Accuracy F1 Accuracy F1 Accuracy F
Dataset: IntentChat
SAID (BERT) 52.73 58.49 63.75 67.80 69.18 72.54 74.05 76.
+ QueryAdapt 67.62 69.89 73.07 75.92 75.03 77.88 72.08 75.
Improvement 28.24% 19.49% 14.62% 11.98% 8.46% 7.36% -2.66% -0.89%
SAID (DistilBERT) 50.17 53.41 56.24 60.38 59.66 63.87 65.94 69.
+ QueryAdapt 60.76 63.81 57.12 62.34 60.01 62.93 63.22 67.
Improvement 21.11% 19.47% 1.56% 3.25% 0.59% -1.47% -4.12% -2.82%
SAID (ALBERT) 49.51 50.36 50.62 54.83 61.27 64.22 57.32 61.
+ QueryAdapt 57.98 61.49 56.47 60.22 58.27 62.76 62.65 66.
Improvement 17.11% 22.10% 11.56% 9.83% -4.90% -2.27% 9.30% 8.69%
Dataset: WildChat
SAID (BERT) 50.03 53.05 51.20 54.73 57.17 63.01 56.95 62.
+ QueryAdapt 56.58 61.64 58.08 63.75 60.11 64.01 65.18 68.
Improvement 13.09% 16.19% 13.44% 16.48% 5.14% 1.59% 14.45% 9.45%
SAID (DistilBERT) 34.38 33.29 45.90 50.14 50.43 54.88 49.48 53.
+ QueryAdapt 47.52 48.89 47.41 50.78 50.84 56.86 56.39 62.
Improvement 38.22% 46.86% 3.29% 1.28% 0.81% 3.61% 13.97% 15.33%
SAID (ALBERT) 43.30 48.47 52.03 56.97 58.75 62.89 59.03 64.
+ QueryAdapt 42.42 46.35 53.12 57.68 60.76 65.39 62.30 65.
Improvement -2.03% -4.37% 2.09% 1.25% 3.42% 3.98% 5.54% 2.59%
```
ear and QueryAdapt strategies lies in whether the learn-
able weights for generating intent-specific relation tokens
are query dependent. As seen from Table 7, QueryAdapt
achieves at least 3.88% and 7% higher Accuracy than Linear
on the IntentChat and WildChat datasets, respectively. This
result supports our motivation that different user queries
may benefit from pretrained relations to varying degrees
when inferring user intent. Meanwhile, we can find that
Linear significantly outperforms the MLP strategy in most
cases, suggesting that simply introducing non-linearity and
additional parameters does not necessarily lead to improved
model performance.

**5.8 QueryAdapt on Various Backbones (RQ7)**

In this section, we evaluate the plug-and-play capability
of the proposed explicit knowledge transfer strategy across
different PLM-based backbones. Specifically, we adopt three
widely used PLMs, including BERT, DistilBERT and AL-
BERT (as in Section 5.3). The experimental results on both
datasets are reported in Table 8, demonstrating the per-
formance improvements of the QueryAdapt empowered
variant over the corresponding default SAID framework.
These results show that the proposed QueryAdapt strategy
consistently yields significant improvements across various
PLMs that differ substantially in architecture and/or model
size, under most settings.

**5.9 Case Study on Attention (RQ8)**

In this section, we conduct a case study to examine what
the attention weights λqqand λqahave learned using the

```
100 75 50 25 0 25 50 75 100
```
```
75
```
```
50
```
```
25
```
```
0
```
```
25
```
```
50
```
```
75
```
```
100 QQ-dominated
QA-dominated
```
```
Fig. 5: Query embedding visualizations on IntentChat.
```
(^05000) **Text Length 10000 15000 20000
100
101
102
103**
(a) QQ-dominated
(^05000) **Text Length 10000 15000 20000
100
101
102
103**
(b) QA-dominated
Fig. 6: Comparison of query texts between QQ- and QA-
Dominated categories on IntentChat.
IntentChat dataset. First, we divide user queries in the test
set into two categories: QQ-dominated, where the estimated
λqqfor a given user query is greater than λqa, and QA-
dominated, where the opposite holds. Then, we employ
the t-SNE model [46] to project the query text embeddings,


obtained from the fixed PLM encoder fplm(·) in Equation
(5), into a two-dimensional space. The visualization results
are shown in Figure 5. Each dot in the figure represents
a user query, with its color indicating the corresponding
category, i.e., QQ-dominated or QA-dominated. As shown
in the figure, the two categories form distinct clusters with
relatively clear boundaries.
We then compare the query texts between the two cate-
gories to better understand why certain user queries benefit
more from query-query pretrained relations, while others do
not. Specifically, we plot the text length distributions of user
queries from both categories in Figure 6. The figure reveals
that QQ-dominated queries (i.e., λqqis greater) tend to be
longer on average. This may be because longer queries often
contain more diverse and potentially noisy information. In
such cases, it becomes especially important to rely more
on query refinement, which refers to how users refine and
clarify the core purpose of their query (as encoded by the
query-query relation tokens during pretraining), to help
extract key information for accurate intent detection.

## 6 CONCLUSION

In this paper, we propose a novel structure-aware pre-
training framework, SAID, for the few-shot query intent
detection problem. This framework unifies both textual
and relational structure information inherent in conver-
sational systems for model pretraining, marking the first
attempt in this field to the best of our knowledge. SAID
features a flexible relation-aware prompt and a structure-
aware masked language modeling module for pretraining,
along with prompt learning-based task reformulation for
few-shot fine-tuning to facilitate effective transfer learning.
Beyond the task reformulation via relation-aware prompt at
the modeling paradigm level, we further introduce an en-
hanced model named SAID (+QueryAdapt) empowered by
a query-adaptive attention network, which operates at the
relation token level to enable more fine-grained knowledge
transfer from pretraining to downstream few-shot intent
detection. Empirical results show significant performance
improvements over state-of-the-art methods. Furthermore,
our framework can be seamlessly integrated with various
backbone models, demonstrating its versatility and robust-
ness.

## REFERENCES

[1] J. D. M.-W. C. Kenton and L. K. Toutanova, “Bert: Pre-training of
deep bidirectional transformers for language understanding,” in
Proceedings of naacL-HLT, vol. 1. Minneapolis, Minnesota, 2019,
p. 2.
[2] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena,
Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer
learning with a unified text-to-text transformer,” Journal of
Machine Learning Research, vol. 21, no. 140, pp. 1–67, 2020.
[Online]. Available: [http://jmlr.org/papers/v21/20-074.html](http://jmlr.org/papers/v21/20-074.html)
[3] T. B. Brown, “Language models are few-shot learners,” arXiv
preprint arXiv:2005.14165, 2020.
[4] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux,
T. Lacroix, B. Roziere, N. Goyal, E. Hambro, F. Azhar` et al., “Llama:
Open and efficient foundation language models,” arXiv preprint
arXiv:2302.13971, 2023.
[5] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L.
Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al.,
“Gpt-4 technical report,” arXiv preprint arXiv:2303.08774, 2023.

```
[6] G. Team, R. Anil, S. Borgeaud, Y. Wu, J.-B. Alayrac, J. Yu,
R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth et al., “Gemini:
a family of highly capable multimodal models,” arXiv preprint
arXiv:2312.11805, 2023.
[7] D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu,
S. Ma, P. Wang, X. Bi et al., “Deepseek-r1: Incentivizing reason-
ing capability in llms via reinforcement learning,” arXiv preprint
arXiv:2501.12948, 2025.
[8] S. Wu, X. Shen, and R. Xia, “A new dialogue response generation
agent for large language models by asking questions to detect
user’s intentions,” arXiv preprint arXiv:2310.03293, 2023.
[9] C. Qian, B. He, Z. Zhuang, J. Deng, Y. Qin, X. Cong, Z. Zhang,
J. Zhou, Y. Lin, Z. Liu et al., “Tell me more! towards implicit user
intention understanding of language model driven agents,” arXiv
preprint arXiv:2402.09205, 2024.
[10] H. Zhang, Y. Zhang, L.-M. Zhan, J. Chen, G. Shi, X.-M. Wu, and
A. Y. Lam, “Effectiveness of pre-training for few-shot intent classi-
fication,” in Findings of the Association for Computational Linguistics:
EMNLP 2021, 2021, pp. 1114–1120.
[11] E. Shnarch, A. Gera, A. Halfon, L. Dankin, L. Choshen,
R. Aharonov, and N. Slonim, “Cluster & tune: Boost cold start
performance in text classification,” in Proceedings of the 60th Annual
Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers), 2022, pp. 7639–7653.
[12] H. Zhang, H. Liang, Y. Zhang, L.-M. Zhan, X.-M. Wu, X. Lu, and
A. Lam, “Fine-tuning pre-trained language models for few-shot
intent detection: Supervised pre-training and isotropization,” in
Proceedings of the 2022 Conference of the North American Chapter
of the Association for Computational Linguistics: Human Language
Technologies, 2022, pp. 532–542.
[13] J. Zhang, T. Bui, S. Yoon, X. Chen, Z. Liu, C. Xia, Q. H. Tran,
W. Chang, and S. Y. Philip, “Few-shot intent detection via con-
trastive pre-training and fine-tuning,” in Proceedings of the 2021
Conference on Empirical Methods in Natural Language Processing,
2021, pp. 1906–1912.
[14] Y. Liu, “Roberta: A robustly optimized bert pretraining approach,”
arXiv preprint arXiv:1907.11692, 2019.
[15] S. Parikh, M. Tiwari, P. Tumbade, and Q. Vohra, “Exploring zero
and few-shot techniques for intent classification,” in Proceedings
of the 61st Annual Meeting of the Association for Computational
Linguistics (Volume 5: Industry Track), 2023, pp. 744–751.
[16] T. Gao, X. Yao, and D. Chen, “Simcse: Simple contrastive learn-
ing of sentence embeddings,” in EMNLP 2021-2021 Conference on
Empirical Methods in Natural Language Processing, Proceedings, 2021.
[17] I. Casanueva, T. Temcinas, D. Gerz, M. Henderson, and I. Vuliˇ c, ́
“Efficient intent detection with dual sentence encoders,” in Pro-
ceedings of the 2nd Workshop on Natural Language Processing for
Conversational AI, 2020, pp. 38–45.
[18] S. Mehri, M. Eric, and D. Hakkani-Tur, “Dialoglue: A natural
language understanding benchmark for task-oriented dialogue,”
arXiv preprint arXiv:2009.13570, 2020.
[19] H. Zhang, H. Liang, L.-M. Zhan, X.-M. Wu, and A. Y. Lam, “Revisit
few-shot intent classification with plms: Direct fine-tuning vs. con-
tinual pre-training,” in Findings of the Association for Computational
Linguistics: ACL 2023, 2023, pp. 11 105–11 121.
[20] L. Zhang, C. Long, and G. Cong, “Region embedding with intra
and inter-view contrastive learning,” IEEE Transactions on Knowl-
edge and Data Engineering, vol. 35, no. 9, pp. 9031–9036, 2022.
[21] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple
framework for contrastive learning of visual representations,” in
International conference on machine learning. PMLR, 2020, pp. 1597–
1607.
[22] J. Zhang, K. Hashimoto, W. Liu, C.-S. Wu, Y. Wan, S. Y. Philip,
R. Socher, and C. Xiong, “Discriminative nearest neighbor few-
shot intent detection by transferring natural language inference,”
in Proceedings of the 2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), 2020, pp. 5064–5082.
[23] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena,
Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer
learning with a unified text-to-text transformer,” Journal of machine
learning research, vol. 21, no. 140, pp. 1–67, 2020.
[24] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever et al.,
“Language models are unsupervised multitask learners,” OpenAI
blog, vol. 1, no. 8, p. 9, 2019.
[25] J. Davison, J. Feldman, and A. M. Rush, “Commonsense knowl-
edge mining from pretrained models,” in Proceedings of the 2019
conference on empirical methods in natural language processing and
```

the 9th international joint conference on natural language processing
(EMNLP-IJCNLP), 2019, pp. 1173–1178.
[26] Y. Gu, X. Han, Z. Liu, and M. Huang, “Ppt: Pre-trained prompt
tuning for few-shot learning,” in Proceedings of the 60th Annual
Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers), 2022, pp. 8410–8423.
[27] N. V. Nayak, P. Yu, and S. Bach, “Learning to compose soft
prompts for compositional zero-shot learning,” in The Eleventh
International Conference on Learning Representations.
[28] T. Schick and H. Schutze, “Exploiting cloze-questions for few-shot ̈
text classification and natural language inference,” in Proceedings
of the 16th Conference of the European Chapter of the Association for
Computational Linguistics: Main Volume, 2021, pp. 255–269.
[29] B. Lester, R. Al-Rfou, and N. Constant, “The power of scale
for parameter-efficient prompt tuning,” in Proceedings of the 2021
Conference on Empirical Methods in Natural Language Processing,
2021, pp. 3045–3059.
[30] X. Liu, K. Ji, Y. Fu, W. Tam, Z. Du, Z. Yang, and J. Tang, “P-tuning:
Prompt tuning can be comparable to fine-tuning across scales and
tasks,” in Proceedings of the 60th Annual Meeting of the Association for
Computational Linguistics (Volume 2: Short Papers), 2022, pp. 61–68.
[31] P. Liu, W. Yuan, J. Fu, Z. Jiang, H. Hayashi, and G. Neubig, “Pre-
train, prompt, and predict: A systematic survey of prompting
methods in natural language processing,” ACM Computing Sur-
veys, vol. 55, no. 9, pp. 1–35, 2023.
[32] P. Sahoo, A. K. Singh, S. Saha, V. Jain, S. Mondal, and
A. Chadha, “A systematic survey of prompt engineering in large
language models: Techniques and applications,” arXiv preprint
arXiv:2402.07927, 2024.
[33] M. Jia, L. Tang, B.-C. Chen, C. Cardie, S. Belongie, B. Hariharan,
and S.-N. Lim, “Visual prompt tuning,” in European Conference on
Computer Vision. Springer, 2022, pp. 709–727.
[34] H. Bahng, A. Jahanian, S. Sankaranarayanan, and P. Isola, “Explor-
ing visual prompts for adapting large-scale models,” arXiv preprint
arXiv:2203.17274, 2022.
[35] Z. Liu, X. Yu, Y. Fang, and X. Zhang, “Graphprompt: Unifying
pre-training and downstream tasks for graph neural networks,”
in Proceedings of the ACM Web Conference 2023, 2023, pp. 417–428.
[36] M. Sun, K. Zhou, X. He, Y. Wang, and X. Wang, “Gppt: Graph pre-
training and prompt tuning to generalize graph neural networks,”
in Proceedings of the 28th ACM SIGKDD Conference on Knowledge
Discovery and Data Mining, 2022, pp. 1717–1727.
[37] M. Henderson, I. Casanueva, N. Mrkˇsi ́c, P.-H. Su, T.-H. Wen,
and I. Vulic, “Convert: Efficient and accurate conversational rep- ́
resentations from transformers,” in Findings of the Association for
Computational Linguistics: EMNLP 2020, 2020, pp. 2161–2174.
[38] Y. Zhang, S. Sun, M. Galley, Y.-C. Chen, C. Brockett, X. Gao, J. Gao,
J. Liu, and W. B. Dolan, “Dialogpt: Large-scale generative pre-
training for conversational response generation,” in Proceedings
of the 58th Annual Meeting of the Association for Computational
Linguistics: System Demonstrations, 2020, pp. 270–278.
[39] C.-S. Wu, S. C. Hoi, R. Socher, and C. Xiong, “Tod-bert: Pre-trained
natural language understanding for task-oriented dialogue,” in
Proceedings of the 2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), 2020, pp. 917–929.
[40] W. Zhao, X. Ren, J. Hessel, C. Cardie, Y. Choi, and Y. Deng,
“Wildchat: 1m chatgpt interaction logs in the wild,” in The Twelfth
International Conference on Learning Representations.
[41] Y. Zhang, C. Yuan, X. Wang, Z. Bai, and Y. Liu, “Learn to adapt
for generalized zero-shot text classification,” in Proceedings of the
60th Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), 2022, pp. 517–527.
[42] V. Sanh, “Distilbert, a distilled version of bert: Smaller, faster,
cheaper and lighter,” arXiv preprint arXiv:1910.01108, 2019.
[43] Z. Lan, “Albert: A lite bert for self-supervised learning of language
representations,” arXiv preprint arXiv:1909.11942, 2019.
[44] L. Tunstall, N. Reimers, U. E. S. Jo, L. Bates, D. Korat,
M. Wasserblat, and O. Pereg, “Efficient few-shot learning without
prompts,” arXiv preprint arXiv:2209.11055, 2022.
[45] G. Liu, L. Zhang, and J. Wu, “Beyond similarity: Relation-based
collaborative filtering,” IEEE Transactions on Knowledge and Data
Engineering, vol. 35, no. 1, pp. 128–140, 2021.
[46] L. Van der Maaten and G. Hinton, “Visualizing data using t-sne,”
Journal of Machine Learning Research, vol. 9, no. 2605, pp. 2579–2605,
2008.


