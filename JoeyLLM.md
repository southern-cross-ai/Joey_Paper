# Lessons from Building a Foundational Model for Sovereign AI in Australia

## Abstract

*This paper reflects on a year of experimentation and discovery by Southern Cross AI, an open-source community initiative working toward sovereign AI capability in Australia. The project began with a simple goal: fine-tune an existing large language model on Australian data to make it more locally relevant. But as we gathered and curated large volumes of regional content from Australian books to government records and web archives we discovered that data alone wasn’t enough. Fine-tuning could not address deeper limitations in architecture, scalability, and cultural alignment. This led to a major shift: the development of BabyJoey, and now JoeyLLM, a foundational model built from the ground up for Australian use cases. In this paper, we share the technical and organisational lessons learned from our early prototypes, the challenges of infrastructure and transformer-level understanding, and the realities of coordinating an open community and Australian university students around complex, resource-intensive work. While JoeyLLM remains in its early stages, our experience underscores why foundational model development, not just local adaptation, is essential for building trustworthy, sovereign AI systems tailored to national contexts.*

## 1 Introduction

1.2 Purpose of This Paper
This paper presents a reflective account of Southern Cross AI’s first year developing JoeyLLM, an open-source foundational language model designed specifically for Australia. Rather than showcasing a completed product, it documents the transition from an initially modest goal of fine-tuning an existing model to the more ambitious and technically complex effort of building a sovereign AI system from the ground up. We share lessons from that transition, discuss the challenges encountered, and offer insights that may help others working on similar national-scale AI projects.

1.2 Background: Southern Cross AI and the Initial Fine-Tuning Plan
Southern Cross AI began as a civic-minded, open-source initiative with a practical goal: to create a language model that could better reflect Australian language, culture, and institutions. Our initial plan was to fine-tune an existing open-weight model, such as LLaMA or Mistral, using a curated corpus of Australian data. The assumption was simple: if we gathered enough local content, we could produce a version of a model that “sounded Australian” without needing to build from scratch.
To support this, we undertook extensive data collection and cleaning efforts. These included:
Literary and public domain texts: Project Gutenberg Australia and custom-chunked variants
Government and academic content: Wiki-Australia, Australian Women's Register (AWR), ArXiv Australia, and Hansard-style transcripts
Corpora of spoken and regional English: ICE-AUS, COOEE, CoANZSE, and the Australian Corpus of English (ACE)
Informal and social media content: Reddit posts, YouTube comments, Twitter datasets (including election-related tweets and regional conversations)
Broadcast and conversational sources: Australian Radio Talkback Corpus (ART)
Filtered web-scale content: Subsets of Hugging Face’s FineWeb and Common Crawl restricted to .au domains
These datasets represented a diverse and uniquely Australian linguistic landscape. However, we soon discovered that even large volumes of high-quality, regionally grounded data were insufficient. The limitations were architectural, not just informational.
Our early experiments included fine-tuning Mistral-7B using Hugging Face tools primarily to test tooling and prompt alignment, but these remained peripheral. The real turning point was the creation of BabyJoey, our first serious attempt to build a foundational model. Based on a GPT-2-scale transformer with six decoder blocks, BabyJoey was trained on our curated corpus, including the Gutenberg Australia dataset.
Despite its modest scale, BabyJoey's tendency to speak with an outdated Australian tone, almost like a character from a 1960s bush novel or historical radio play, BabyJoey struggled to generate coherent responses. The limited model size meant it often rambled, repeated itself, or drifted off-topic, sounding more like an eccentric period drama than a usable assistant. This wasn’t unexpected. Small models rarely generalise well; they lack the depth and contextual capacity to produce consistent or meaningful outputs. But it confirmed firsthand what we suspected: solving for cultural alignment, scalability, and efficiency would require complete control over the architecture, training process, and deployment environment, not just the data.
This marked a key inflection point for Southern Cross AI: a shift away from adapting pre-trained models and toward building foundational systems from the ground up for Australian needs.

1.3 Why Sovereign AI Matters for Australia
Large language models are no longer experimental they are becoming embedded in public services, policy delivery, education, legal assistance, and communication. Most of these models are developed overseas, governed by commercial terms, and shaped by cultural and political norms that don’t necessarily reflect Australian values, legal frameworks, or regional needs.
For a country like Australia, which relies heavily on imported digital infrastructure, the risks of dependency are real. If access to foreign AI models is throttled, restricted, or misaligned with national priorities, entire public systems could be affected. Sovereign AI is not about isolation, it’s about resilience. It means having the capacity to audit, adapt, and deploy models within local infrastructure, guided by local governance.
1.4 What This Paper Reflects On
This paper is not a technical blueprint or a product showcase. Instead, it’s a documentation of what was learned technically, organizationally, and socially in the process of building Australia’s first open-source foundational model. We reflect on:
Why fine-tuning alone proved insufficient
The infrastructure and architectural challenges of scaling
The early lessons from building BabyJoey
The complexities of coordinating a distributed, volunteer-based team, including university contributors
The rationale for developing JoeyLLM as a foundational model
What lies ahead as the model scales toward national relevance
Through these reflections, we aim to contribute to a growing global conversation about sovereign AI not as a talking point, but as a lived technical and civic practice.

## 2. Initial Approach and Assumptions
2.1 Fine-Tuning as the Starting Point
At the outset, Southern Cross AI operated under a common assumption that, with enough targeted data, an existing large language model could be fine-tuned to produce culturally relevant and locally grounded outputs for Australian use cases. The early plan was not to build a model from scratch, but to take advantage of open-weight models like LLaMA or Mistral, and adapt them using standard instruction tuning techniques and region-specific corpora. This was seen as the fastest, most pragmatic route to producing an "Australian" LLM suitable for experimentation, downstream integration, and public sector applications.
This plan also made sense for a newly forming community. Many of our earliest contributors, including academic researchers, independent AI practitioners, and student volunteers, saw fine-tuning as a manageable entry point. At that stage, we had not yet secured interest from major government or industry stakeholders. The work was being done by those already close to the open-source ecosystem: people who could quickly get involved and test ideas using the available tools.
2.2 Data Collection Efforts
With that goal in mind, we focused initially on curating high-quality Australian datasets. This effort went far beyond scraping headlines or pulling PDFs. Instead, we organised data around a mix of formal, institutional, conversational, and regional content types. Key sources included:
Project Gutenberg Australia, chunked into different token lengths to test formatting and memory efficiency
Wiki-Australia, AWR, and Hansard-style content to reflect governmental and academic language
ICE-AUS, COOEE, CoANZSE, and ACE, which provided Australian spoken English in formal and informal contexts
Reddit, YouTube comments, and Twitter corpora to capture everyday phrasing, slang, and informal discourse
Australian Radio Talkback Corpus (ART), offering real-world audio transcripts
FineWeb and Common Crawl subsets, filtered to include only .au domains or Australia-relevant topics
This data was largely gathered and prepared through open collaboration. Students, hobbyists, and volunteers from our Discord community helped source, clean, tag, and structure the data, often working across time zones and institutions. The shared goal of creating something truly Australian brought together contributors from ANU, the open-source community, and experienced developers from the AI field. Even at this early stage, community participation was central to our progress.
2.3 Early Expectations vs. Reality
The early expectation was that this combination of high-volume, culturally specific data and instruction tuning would produce a sufficiently aligned model with relatively little architectural intervention. In other words, we thought the model would “sound Australian” if trained on Australian content.
Early Observations from Fine-Tuning
As we began evaluating outputs from our fine-tuned Mistral models, it became clear that our initial assumptions were incomplete. The models exhibited issues with coherence, tone drift, and generalization.
Although the fine-tunes produced technically fluent outputs, the models often reverted to stylistic patterns shaped by their original training data. This commonly manifested as caricatured Australian personas—reminiscent of a stereotypical “Crocodile Dundee” figure. The tone leaned exaggerated and boastful, echoing American cultural tropes, and missed the self-deprecating nuance characteristic of Australian communication.
This highlighted a deeper challenge: capturing the subtle cultural cues and linguistic intricacies of Australian English. While the models could mimic local slang or phrasing, they failed to authentically reflect the broader cultural context—underscoring the limits of fine-tuning alone for achieving cultural fidelity.
Even when outputs were factually grounded, they lacked the flexibility, nuance, and relevance required for real-world deployment. Most importantly, we found that cultural alignment could not be achieved through data alone. Architecture, training dynamics, and tokenization strategy played a more decisive role than we had initially anticipated.
This realization prompted a strategic pivot—from adapting existing models to building our own. While not all contributors agreed on this shift, the divergence in viewpoints reflected strong engagement: people cared not just about the technology, but about the direction the project should take.
2.4 Community Formation and Early Collaboration
From the beginning, Southern Cross AI was envisioned not just as a model-building project, but as a platform for civic collaboration. The idea of a sovereign language model for Australia resonated across different groups technologists, educators, AI researchers, and public-minded citizens. We aimed to bring together voices from government, academia, startups, and hobbyist circles.
In practice, early buy-in skewed heavily toward academia and hobbyist communities. Researchers from the Australian National University and a number of experienced open-source developers joined quickly. Government and business interest was more cautious, largely due to the experimental nature of the project and the lack of a concrete model to demonstrate at the time.
The community grew organically through Discord, GitHub, and shared datasets. Students and volunteers led key efforts in dataset collection, formatting, and early prompt engineering. Their work formed the backbone of the corpus that would later be used to train BabyJoey. This was not just a technical achievement it was a community milestone.
That said, coordination wasn't always easy. As the project shifted away from fine-tuning toward full foundational model development, viewpoints diverged. Some contributors saw the pivot as visionary, while others saw it as overly ambitious. These tensions reflected broader debates in the open-source AI world: about scale, centralization, and the meaning of "sovereign AI."
Despite these differences, the community delivered. It built the early infrastructure, shaped the data, and grounded the model’s identity in collective effort. While far from perfect, it demonstrated that a civic-minded open-source initiative could contribute meaningfully to one of the most advanced areas of machine learning and do so on its own terms.

## 3. Prototype: Baby Joey
3.1 Technical Overview
BabyJoey was the first foundational model trained by Southern Cross AI a deliberately small-scale prototype designed to validate our training pipeline and test end-to-end processes for Australian-specific LLM development. The model architecture was intentionally minimal:
Component
Value
Layers
6
Hidden size
512
Attention heads
8
Sequence length
512
Vocabulary size
50,000
Total parameters
~64 million

BabyJoey was implemented using the PyTorch library and trained on a single GPU using local compute. The dataset consisted solely of a custom-processed version of Project Gutenberg Australia. For tokenization, we used OpenAI’s cl100k_base vocabulary via the tiktoken library. While this tokenizer was not trained on Australian-specific content, it allowed us to rapidly test our training pipeline using a well-optimized subword vocabulary. Preprocessing included de-duplication, formatting, and length-based chunking.
The goal was not model performance, but infrastructure validation. We set out to confirm that we could ingest data, tokenize effectively, configure a working architecture, run a complete training cycle, and generate outputs entirely on Australian-controlled infrastructure.

3.2 Model Behavior and Limitations
As expected for a 64M parameter model, BabyJoey struggled to produce coherent or consistent outputs. While it successfully generated grammatically correct phrases and occasional factoids, it often rambled, repeated itself, or lost track of context after just a few lines.
Most notably, the model took on a distinctly old-fashioned tone more like a character from a 1960s bush novel than a contemporary assistant. This was unsurprising: its training data skewed heavily toward older, copyright-free literature, which shaped its vocabulary and style. It often responded in poetic or narrative form, even when prompted with factual or instructional questions.
While charming in tone, it clearly lacked the capacity for real-world use. There were no meaningful reasoning capabilities, and the model failed at even simple question answering tasks outside its narrow training window.
3.3 Lessons Learned
Despite its limitations, BabyJoey gave us critical insights:
Small models can’t align: With only 64M parameters, BabyJoey lacked the capacity to generalize or hold context, even across short prompts. This confirmed that alignment is not just a data problem it’s a scale problem.
Training from scratch is doable: Even on minimal infrastructure, we were able to train a working LLM using local code, community-collected data, and an open-source stack.
Tone reflects data sometimes too well: The outdated, overly formal voice taught us how strongly style is influenced by corpus era and composition. It emphasized the need for more contemporary, diverse sources.
Pipeline maturity matters: From tokenization to logging and checkpointing, the process helped us debug critical parts of the infrastructure we’d later need to scale up.
3.4 The Turning Point
More than anything, BabyJoey proved that fine-tuning alone was not the answer. Even with carefully curated data, the architectural and scale limitations of small models couldn’t be ignored. The experience validated our decision to go deeper: to understand transformer architecture from the ground up, and to build the capacity both technical and organizational to train larger, more useful models.
That realization marked the transition from BabyJoey to JoeyLLM a multi-phase effort aimed at creating an Australian foundational model capable of real alignment, scalability, and deployment in public-interest contexts.

## 4. Shift to Foundational Model Development
4.1 Realizing the Limitations of Fine-Tuning
Our original plan to fine-tune existing open-weight models using Australian-specific data was grounded in practicality. Fine-tuning seemed more accessible, affordable, and immediately useful. But after early experiments with Mistral-7B and evaluation of community efforts built on LLaMA and similar models, we encountered a consistent pattern: fine-tuning could localize style, but not shift underlying assumptions or limitations.
Models trained elsewhere came with invisible baggage tokenization choices, training data biases, and architectural tradeoffs baked in by external priorities. Even when we applied high-quality, well-structured Australian data, the resulting models retained artifacts of their origin: foreign political assumptions, misaligned tone, and a lack of context for local institutions or norms. It became clear that you can’t align a model to your values if you don’t control how it was built.
4.2 Architecture Constraints
As we explored what it would take to scale up, we realized that architecture matters just as much as data. Models like Mistral and DeepSeekMoE demonstrated how sparse activation, attention optimization, and high-efficiency routing architectures could drastically shift what was possible even with limited compute.
By contrast, the models we were working with (both open and experimental) were rigid: built for general-purpose English, with no way to adapt their core internals without breaking compatibility or retraining from scratch.
Our experience with BabyJoey reinforced this lesson. While the model was small, the process of building and training it helped us understand how every choice layer depth, hidden size, attention heads, token length impacted behavior, training time, and generalization. These weren’t just technical tweaks they were foundational design decisions we needed to own.
4.3 Deployment and Scaling Problems
We also began to think more seriously about how these models would be used not just built. Many potential users in education, health, law, and public sector contexts needed models that could run offline, in secure or bandwidth-constrained environments, or on modest hardware.
But most large-scale open models were designed for cloud deployment in inference-optimized clusters not for lightweight, regionally adaptable inference. Fine-tuning couldn’t solve this either. Without the ability to design the model for deployment from the start, we risked building a system that couldn’t run in the places where it mattered most.
Foundational model development gave us the opportunity to shape not just the model’s outputs, but its compute profile, quantization pathway, and compatibility with edge and offline scenarios.
4.4 Decision to Pursue a Full Foundational Model
These limitations, taken together, led to a decisive pivot: we would build JoeyLLM from the ground up as an Australian foundational model not just a fine-tune of someone else’s.
This shift was not just technical it was strategic. A foundational model would give us:
Architectural control, enabling us to experiment with scaling, routing, and efficient training
Deployment flexibility, allowing us to target use in constrained public environments
Cultural and civic alignment, grounded in local data, norms, and needs
Governance transparency, so that the full training and tuning process could be audited, adapted, and improved by the community
It also set a higher bar for sustainability. Foundational models are harder to build but they offer long-term sovereignty and adaptability. They become infrastructure, not just tools.
With that shift, the JoeyLLM project took on new urgency and scope. We were no longer just trying to localize a model we were trying to build national capability.

## 5. Infrastructure and Technical Challenges
5.1 Understanding Transformer Internals
Building BabyJoey gave us hands-on exposure to the internals of transformer-based architectures. While much of the current open-source ecosystem abstracts these details, our goal was to understand how the system works beneath the surface from attention mechanisms and layer normalization to optimizer scheduling and token routing.
We worked directly with low-level components in PyTorch, deliberately avoiding overreliance on high-level training frameworks. This helped us map out the full training stack: not just what goes into the model, but how data flows through it, how memory bottlenecks arise, and how inference behavior emerges from weight configuration.
This process allowed us to build technical literacy within the team and among students and volunteers about what actually makes a language model tick. It was an essential step in building the capacity to scale up.

5.2 Platform and Infrastructure Realities
Southern Cross AI is an open-source initiative, not a billion-dollar lab. That means we operate with the constraints of shared infrastructure, academic compute quotas, donated GPU time, and locally available hardware. This shaped every architectural and training decision.
Our environment is primarily Linux-based, with experiments run across a mix of local servers, university research clusters, and limited cloud credits. Many contributors train or fine-tune models on consumer-grade hardware including older NVIDIA cards, laptops, or even CPU-only rigs for preprocessing and evaluation.
These limitations forced us to become efficient early. We adopted lightweight monitoring tools, built modular training loops, and worked to keep our data pipelines GPU-aware and fail-safe. Every optimization had to be justified; every gigabyte mattered. It also meant that decisions made now would have lasting consequences on whether JoeyLLM could actually be used across the country in varied, resource-constrained environments.

5.3 Preparing for Future Architectures: Mixture of Experts and Beyond
One of our long-term goals is to scale JoeyLLM using Mixture of Experts (MoE) architectures, inspired by models like DeepSeekMoE and Mixtral. These models activate only a subset of their parameters during inference, allowing for efficient scaling without full-compute overhead.
However, preparing for MoE requires foundational understanding: routing mechanisms, expert balancing, and auxiliary loss management. We are currently studying sparse activation strategies and auxiliary-loss-free gating mechanisms, with the intention of implementing them in future JoeyLLM variants.
We’ve also explored the tooling that supports these approaches such as DeepSpeed, FlashAttention, and fused optimizer libraries which are necessary for managing memory and performance at scale. But integrating these into a model that can still run on Australian hardware, or be deployed offline, remains an ongoing challenge.

5.4 Emphasis on Reproducibility and Edge Readiness
Given our focus on public interest applications particularly in education, healthcare, and legal services reproducibility and edge-readiness are non-negotiable.
Our models must be:
Reproducible, with version-controlled training scripts, open datasets, and fully documented experiments
Portable, with support for Docker-based inference environments, ONNX exports, and quantized model variants
Deployable, in secure or disconnected environments such as schools, local councils, or rural health clinics
We’ve begun building internal tooling (under the “ModelWorks” framework) that ensures all stages of the JoeyLLM pipeline from preprocessing to evaluation can be run with minimal dependencies and full traceability. Our long-term vision is to support small and mid-scale deployments, where sovereignty isn’t just about model weights, but about control over how and where a model is used.

## 6. Data Strategy and Cultural Grounding
6.1 Curation of Australian Data Sources
From the beginning, Southern Cross AI took a deliberate, bottom-up approach to data collection. Rather than scraping indiscriminately from the internet, we focused on curating high-quality, Australian-specific datasets that reflect the nation’s linguistic, institutional, and cultural landscape.
Key sources included:
Project Gutenberg Australia, including Australian authors, processed into chunked formats for different context lengths
Government publications, including open-access reports, parliamentary records, policy briefs, and legislation
Australian academic corpora, such as ICE-AUS, COOEE, ACE, and CoANZSE, which reflect formal and informal regional English
Social and conversational datasets, including filtered Reddit and Twitter posts, YouTube comments, and talkback radio transcripts
Specialist domains, such as the Australian Women's Register, health documents, and education materials aligned with the national curriculum
Web-scale datasets, like Hugging Face’s FineWeb and Common Crawl, filtered specifically to .au domains and regionally relevant sites
This corpus reflects a wide cross-section of Australian life from formal parliamentary language to everyday online discourse.

6.2 Challenges of Quantity and Quality
Despite the breadth of content, two consistent challenges emerged: data quantity and data quality.
In terms of quantity, Australian web and literature content is simply limited compared to global-scale datasets used in commercial LLMs. The total volume of publicly usable, copyright-safe Australian content remains small by comparison even with heavy scraping and filtering. Much of what exists is domain-specific, fragmented, or dated.
In terms of quality, many texts especially in the public domain carry stylistic, cultural, or linguistic quirks that skew the model's tone and generalization. Early experiments with BabyJoey revealed that a model trained solely on older literature would adopt an archaic, overly formal voice. Modernity, diversity, and inclusivity require more than quantity they require representative sampling from voices that are current, grounded, and broad-based.
We also encountered technical challenges: formatting inconsistencies, metadata loss, data duplication, and the need for ethical filtering to avoid inappropriate or biased material. Each dataset required hand-tuning and often community review before it was integrated into training pipelines.

6.3 Handling Indigenous Data and Ethical Governance
One of the most sensitive areas in our data strategy is the treatment of Indigenous language, knowledge, and cultural materials.
Southern Cross AI takes a firm stance: no Indigenous data is included in any training corpus without explicit permission and community-defined terms of use. We treat Indigenous data as governed by its own sovereignty not as a subset of “Australian data” to be included by default.
This means that:
Any content involving First Nations voices must be reviewed by relevant cultural custodians
Usage must be aligned with community goals such as language preservation, health equity, or cultural education not generic training performance
Protocols for consent, attribution, and ongoing governance are respected
In some cases, we’ve engaged with Indigenous advisors and researchers to explore co-creation of data resources. In other cases, we’ve deliberately chosen not to include potentially sensitive materials, recognizing that absence is sometimes more ethical than inclusion without trust.

6.4 Why Alignment Is About More Than Data Inclusion
One of the most important lessons we’ve learned is that alignment cannot be solved through data alone. Including Australian data even in large volumes does not guarantee a model that behaves appropriately, understands local norms, or communicates effectively in civic contexts.
Alignment also depends on:
Model architecture and training scale, which affect generalization and tone
Instruction tuning strategies, which shape how the model interprets prompts and interacts with users
Governance structures, which determine who decides what the model should do and not do
Evaluation processes, including red-teaming, community testing, and public sector feedback
In short, cultural alignment is not a dataset it’s a process, one that spans model design, training, fine-tuning, evaluation, and long-term deployment.

## 7. Community Engagement and Organizational Learning
7.1 Southern Cross AI’s Open-Source Structure
Southern Cross AI was founded not just as a technical project, but as a civic infrastructure experiment a space where researchers, developers, public servants, and everyday citizens could work together on a sovereign AI project. From the beginning, the initiative was organized around open-source principles: transparent repositories, open documentation, public Discord discussions, and an open-door contribution model.
Rather than operating under a closed research lab or a single institutional owner, the community grew through shared goals: build something useful, Australian, and open. This model attracted contributors across academic institutions (particularly ANU), hobbyist AI builders, open-source veterans, students, and long-time developers. Early traction came not from formal recruitment but from shared purpose.
7.2 What Worked and What Didn’t
The open model allowed for fast iteration in some areas, especially during the early stages of dataset collection, exploratory training, and tooling validation. Students and volunteers were able to jump in, contribute to preprocessing, test tokenizers, clean datasets, and explore prompt engineering methods. These loosely coordinated sprints generated meaningful progress particularly during BabyJoey’s development and the curation of initial corpora.
However, coordination wasn’t always smooth. As the project’s scope expanded from fine-tuning to foundational modeling the complexity outpaced the informal structure. Some contributors were more interested in use-case development or deployment; others focused solely on technical research. Expectations diverged, and without clear planning rituals or roadmap alignment, some efforts ran in parallel without integration. The lack of centralized decision-making occasionally slowed progress or duplicated work.
There were also drops in momentum, particularly when contributors became unclear about how or where to contribute. Without defined scopes, role rotation, or structured onboarding, newer participants sometimes felt unsure about how to participate meaningfully.
7.3 The Importance of Roles, Rituals, and Communication
Over time, we learned that open collaboration still requires structure. Roles even informal ones matter. Whether it’s a lead on dataset stewardship, a merge manager for the model codebase, or a student responsible for tokenizer testing, assigning clear accountability improved both productivity and morale.
Simple rituals such as weekly check-ins, issue triage on GitHub, and sprint-style roadmapping made a substantial difference in keeping the community aligned. When communication channels were active and scoped around specific tasks, participation surged. When updates were infrequent or abstract, engagement dropped.
In an academic or civic environment where contributors juggle other responsibilities, clarity, purpose, and feedback loops proved to be more valuable than unrestricted openness.
7.4 Balancing Openness with Execution
Southern Cross AI remains committed to open-source values, but has come to understand that openness must be designed not assumed. We are continuing to refine governance models that preserve transparency and participation while also ensuring that deadlines are met, architecture decisions are tracked, and technical debt is managed.
We don’t view structure as a compromise. Instead, we see it as a precondition for scale. A project as complex as building a sovereign LLM cannot rely on spontaneous alignment it needs scaffolding that supports collaborative execution without becoming bureaucratic.
As we move forward with JoeyLLM, we are experimenting with rotating leads, modular project scopes, and contributor mentorship to better balance openness with sustained technical delivery. These organizational lessons are as critical as the model itself because how we build JoeyLLM matters as much as what we build.

## 8. Lessons Learned
After a year of experimentation, community building, and foundational model development, several core insights have emerged from the JoeyLLM project. These lessons are not just technical they span architecture, governance, data strategy, and collaboration. They reflect the complex reality of building sovereign AI in practice, not just in theory.
8.1 Data Alone Isn’t Enough
One of the earliest realizations was that local data, no matter how well-curated, cannot compensate for architectural and scale limitations. While culturally aligned datasets are essential for downstream tuning and evaluation, they don’t fix the deeper assumptions baked into a model’s design. Our experiments with BabyJoey and Mistral fine-tunes showed that even models trained on Australian content continued to behave in ways shaped by their original structure and upstream data bias. Cultural alignment cannot be patched in it must be designed for.
8.2 Architectural Control Is Necessary
True sovereignty means more than having access to weights it means having the ability to design, modify, and scale the architecture itself. Foundational decisions about attention mechanisms, activation sparsity, context length, and tokenization shape how a model behaves long before any prompts are applied. We learned that to build models fit for Australian deployment in education, health, or legal settings we needed control over these internals. Fine-tuning someone else's assumptions wasn't enough.
8.3 Sovereign AI Requires Civic Infrastructure Not Just Code
Building a sovereign model isn’t just a software challenge it’s a nation-building one. From infrastructure (GPUs, training clusters, and compute access) to licensing, governance, and deployment pathways, sovereign AI requires civic-scale thinking. For JoeyLLM to be viable in public sector use cases, it has to run on Australian systems, align with regional law and ethics, and be adaptable to specific environments like schools, courts, and health services. Code alone can’t carry that responsibility an ecosystem must.
8.4 Community Is Critical but Must Be Designed For
Open-source collaboration is powerful, but it doesn't self-organize at scale. What worked in the early days fluid roles, informal sprints, and chat-based planning eventually hit limits. We learned that collaborative execution needs structure: roles, rituals, versioning, scope boundaries, and feedback loops. Importantly, we also saw that the right scaffolding didn’t reduce engagement it enhanced it. When contributors knew how to plug in and why their work mattered, they showed up with purpose.
8.5 Reflections for Other Sovereign AI Efforts
For other communities, countries, or institutions considering their own sovereign AI path, our experience suggests a few guidelines:
Start small but with ownership. Even a 64M parameter model can teach you more than a dozen fine-tunes.
Build openly but architect intentionally. Transparency only works when it’s matched by clear goals and reproducibility.
Engage early even if imperfectly. Community trust and collaboration don’t emerge fully formed; they evolve through action and iteration.
Accept friction. Disagreement, divergence, and technical missteps are part of the process. What matters is how they're handled.
Think beyond models. AI sovereignty includes storage, deployment, governance, and cultural resonance not just training loops.

## 9. Roadmap and Next Steps
The lessons learned through BabyJoey and the early development of JoeyLLM have shaped a clear, pragmatic roadmap for the next phase of the project. This roadmap is not just about scaling parameter counts it’s about scaling capacity, community, and national readiness for sovereign AI systems.
9.1 Goals for Joey (7B–13B Scale)
The next major milestone is the training of JoeyLLM at full scale, targeting a dense transformer architecture in the 7 to 13 billion parameter range. This model will serve as the first high-capacity, open-weight, Australian foundational model suitable for instruction tuning, downstream customization, and real-world inference.
This version of Joey is being designed with:
Longer context windows (4K–16K) for document reasoning
Instruction tuning support, including domain-specific prompt sets
Flexible inference pathways, enabling ONNX export, quantization, and air-gapped deployment
Training optimizations, including the use of fused ops and mixed precision
The model will build on the groundwork laid by BabyJoey, but with dramatically expanded data coverage, updated preprocessing pipelines, and infrastructure-aware architectural decisions.

9.2 Mixture of Experts Exploration
Beyond dense scaling, the team is actively researching Mixture of Experts (MoE) architectures for future JoeyLLM variants. Inspired by recent work from DeepSeek, Mistral, and Google, MoE architectures allow for large model capacity while activating only a fraction of parameters during inference improving compute efficiency without compromising performance.
This line of work includes:
Implementing auxiliary-loss-free gating
Evaluating 2-of-N expert routing strategies
Benchmarking MoE models for edge and cluster deployment trade-offs
The goal is to deliver a Joey variant with the benefits of large-scale representation power while retaining compatibility with modest Australian infrastructure.

9.3 Infrastructure and Deployment Research
Scaling a model is only part of the challenge; running and serving it in the real world is equally important. JoeyLLM is being designed with a strong emphasis on offline and edge readiness, particularly for public sector and regional deployment.
Current efforts focus on:
Dockerized inference environments for use in schools, clinics, and legal services
Quantized versions of the model for use on CPU-only or older GPU hardware
Compatibility with academic clusters and pooled community GPU resources
Secure, reproducible pipelines for training and inference in air-gapped environments
These constraints are not limitations they are design parameters that reflect the environments where sovereignty matters most.

9.4 Sustainable Engagement with Government and Partners
While early interest came mostly from academia and the open-source community, the next stage of JoeyLLM’s development involves building durable relationships with government, industry, and institutional partners. Our goal is not to hand over a finished model, but to co-develop use cases, deployment pathways, and governance models in collaboration with those who will use and regulate these systems.
This includes:
Briefings and pilot discussions with public sector agencies
Engagement with national infrastructure providers (e.g. AWS, NCI, and research networks)
Open frameworks for fine-tuning and adaptation by state and regional bodies
Licensing models that support public use without enabling misuse or misrepresentation
The JoeyLLM initiative is not a vendor it is a civic capability builder.

9.5 Long-Term Vision for Sovereign Capability in Australia
The long-term goal of Southern Cross AI is to establish sovereign AI capability as public infrastructure not as a one-off research project, but as a living ecosystem. That means:
Models that can be fine-tuned and deployed locally
Tools and datasets that reflect Australia’s linguistic, cultural, and civic landscape
Skills development pathways for students, civil servants, and public interest technologists
A governance model rooted in transparency, inclusion, and national resilience
We believe that AI sovereignty is not about isolation, but about agency. JoeyLLM represents a step toward that future a model of what it means to build responsibly, openly, and on our own terms.

## 10. Conclusion
The past year has shown us that building a foundational model is not just a technical task it is a national capability exercise. While fine-tuning can localize outputs, only foundational model development gives the level of control needed to align architecture, data, deployment, and governance with local values and needs. JoeyLLM was born from that realization.
Though our model is still maturing, the process of building BabyJoey and now JoeyLLM has already delivered immense value. It has helped us understand the infrastructure required to train models on Australian data. It has revealed the social complexity of coordinating a civic AI initiative. And it has proven that open, community-driven model development is possible even without the resources of a major AI lab.
We do not claim to have solved all the problems of alignment, scaling, or deployment. But we have made a start one that is grounded in transparency, driven by public interest, and anchored in Australian priorities. What JoeyLLM offers is not just a tool for language generation, but a template for how nations can build their own AI infrastructure from scratch and in the open.
As we continue to grow the project, we extend an invitation not for passive use, but for collaboration. Whether you're in government, academia, research, or the open-source community, the future of sovereign AI will be shaped not by any single model, but by the partnerships we forge around it.
JoeyLLM is our contribution to that future. A model of what can be built together, and on our own terms.
Acknowledgements
The JoeyLLM initiative was made possible through the collaborative efforts of the Southern Cross AI community. We gratefully acknowledge the contributions of student researchers, volunteers, and open-source developers who helped with dataset collection, preprocessing, model experimentation, and governance discussions.
Special thanks to collaborators at the Australian National University (ANU) for early academic engagement, as well as to independent AI practitioners who contributed infrastructure advice, training insights, and feedback throughout the project’s evolution. We also appreciate the support of contributors who offered compute resources, participated in red-teaming, and engaged in community discussions on Discord and GitHub.
We would also like to acknowledge the foundational work of Colin Choat, founder and curator of Project Gutenberg Australia, whose dedication to preserving and sharing Australian literary heritage provided an essential building block for our early dataset development.
Finally, we thank the broader open-source AI ecosystem including the developers of PyTorch, Hugging Face, DeepSpeed, and tokenization libraries for creating the tools that made this project possible. JoeyLLM would not exist without the collective intelligence of the global machine learning community.

## References





