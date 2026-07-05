# Master Report Structure

This file is the updated outline for the main research document.

The main report focuses on:

- problem motivation
- mathematical formulation
- system models
- optimization problems
- proposed methodologies
- algorithmic structure
- modeling assumptions

Empirical validation is reserved for a separate companion results report.

## Proposed title

`Finite-Blocklength Latency Minimization in Uplink and Downlink Systems:
Problem Formulation and Proposed Methods`

## Front matter

### 1. Title page

- title
- authors
- affiliation
- date

### 2. Abstract

The abstract should summarize:

- the uplink and downlink latency-minimization problem
- the finite-blocklength setting
- the shared optimization viewpoint across both links
- the main proposed solution families
- the methodological contributions of the work

### 3. Report scope and contributions

This section should state clearly that the document covers:

- the research problem
- the theoretical setup
- the system assumptions
- the optimization formulations
- the proposed uplink methods
- the proposed downlink methods

It should also state clearly that empirical validation is reported separately.

## Main body

### 4. Introduction

This section should explain:

- why low-latency communication matters
- why finite-blocklength modeling is necessary
- why uplink and downlink should both be studied
- why latency, reliability, power, and blocklength must be optimized jointly
- why both direct optimization and learned precoding strategies are relevant

### 5. Unified notation and common problem setting

This chapter should define all shared symbols and assumptions before splitting
into uplink and downlink.

Include:

- user index
- block index
- payload bits
- blocklength
- reliability target
- symbol rate
- power constraints
- precoder variables
- channel variables
- latency definition

Recommended subsections:

#### 5.1 Common symbols

- symbol glossary
- vector and matrix notation
- uplink/downlink common variables

#### 5.2 Finite-blocklength service model

- service rate under finite blocklength
- feasibility condition
- role of reliability

#### 5.3 Latency definition

- per-user latency
- total latency
- relation between blocklength and delay

#### 5.4 Task variants

Include the two task views used in the work:

- payload-completion setting
- fixed-block-target setting

### 6. Common optimization viewpoint

This section should present the shared conceptual optimization template used
throughout the report.

Include:

- decision variables
- objective
- feasibility constraints
- power constraints
- causality or scheduling structure where relevant

Recommended subsections:

#### 6.1 Decision variables

- transmitted bits per block
- blocklength per block
- precoder per block

#### 6.2 Objective functions

- latency minimization objective
- optional aggregate objective forms used in the methods

#### 6.3 Constraint set

- finite-blocklength rate feasibility
- power limits
- blocklength bounds
- payload or target-bit consistency

### 7. Uplink system model

This chapter should define the uplink side independently and cleanly.

Recommended subsections:

#### 7.1 Physical-layer uplink model

- user transmit structure
- base-station reception model
- channel representation
- noise or interference handling

#### 7.2 Effective uplink finite-blocklength model

- effective achievable service expression
- SNR-based or SINR-based view if needed

#### 7.3 Uplink latency structure

- user-wise service accumulation
- coherence-limited transmission structure
- role of repeated sub-blocks

### 8. Uplink optimization problem

This chapter should move from model to mathematical problem statement.

Recommended subsections:

#### 8.1 Problem statement

- user payload completion under finite-blocklength constraints
- latency objective

#### 8.2 Unknown sub-block count

- why the number of uplink sub-blocks is not known in advance
- why this makes the problem structurally difficult

#### 8.3 Constrained single-sub-block formulation

- one-block feasibility subproblem
- coupling between blocklength and precoder

### 9. Proposed uplink methodologies

This is the main uplink methods chapter.

Recommended subsections:

#### 9.1 Dynamic sub-block construction

- greedy or sequential creation of user sub-blocks
- payload update rule after each committed block
- stopping rule when payload is exhausted

#### 9.2 Single-sub-block constrained solve

- how one candidate block is optimized
- interaction between blocklength choice and precoder choice

#### 9.3 Alternating optimization view

- alternating updates across variables
- reason for using an alternating structure

#### 9.4 Primal-dual or Lagrangian update view

- constrained optimization interpretation
- dual variables
- feasibility enforcement

#### 9.5 Uplink online convergence method

- direct online blockwise optimization
- convergence logic at test time
- role inside the full scheduler

#### 9.6 Uplink Monte Carlo precoder-learning method

- offline training objective
- one-net-per-user structure
- blocklength-aware learned precoder mapping
- use of learned precoder inside the outer uplink scheduling logic

#### 9.7 Relationship between uplink method families

- common components shared by the uplink methods
- where the methods differ
- what is optimized online versus learned offline

### 10. Downlink system model

This chapter should define the downlink side with the interference coupling made
explicit.

Recommended subsections:

#### 10.1 Physical-layer downlink model

- base-station transmission model
- user reception model
- multi-user interference structure

#### 10.2 Interference-coupled finite-blocklength model

- why downlink users cannot be optimized independently within a block
- interference covariance or equivalent coupled service view

#### 10.3 Downlink latency structure

- common block scheduler
- active-user set per block
- accumulated latency across scheduled blocks

### 11. Downlink optimization problem

This chapter should formalize the downlink problem.

Recommended subsections:

#### 11.1 Problem statement

- common-block latency minimization
- per-user service variables
- shared power and interference constraints

#### 11.2 Coupled block optimization difficulty

- why users are jointly coupled through interference
- why blockwise causal scheduling is needed

#### 11.3 Per-block constrained formulation

- blockwise optimization target
- feasibility conditions before bit allocation

### 12. Proposed downlink methodologies

This is the main downlink methods chapter.

Recommended subsections:

#### 12.1 Per-block precoder convergence before allocation

- reason for converging precoders first
- stable interference environment before payload assignment

#### 12.2 Downlink online convergence method

- joint active-user optimization within one block
- convergence logic
- progression from block to block

#### 12.3 Bit allocation after precoder convergence

- separation between beam optimization and bit assignment
- greedy or causal service rule

#### 12.4 Blocklength reduction or refinement logic

- candidate blocklength shrink rule
- feasibility-preserving updates
- causal scheduling logic

#### 12.5 Downlink Monte Carlo precoder-learning method

- offline learning objective
- joint training view
- learned precoder deployment during scheduling

#### 12.6 Downlink architectural variants

Include the conceptual variants only:

- per-user learned precoder structure
- shared base-station learned precoder structure
- jointly coupled versus user-wise service handling

#### 12.7 Relationship between downlink method families

- common scheduler backbone
- difference between online convergence and learned precoder usage
- difference between local and joint learned structures

### 13. Cross-link methodological synthesis

This chapter should connect the uplink and downlink stories at a research level.

Include:

- what is shared between uplink and downlink
- what is fundamentally different
- why uplink admits more user-wise decomposition
- why downlink requires stronger joint treatment
- how the proposed methods reflect those structural differences

Recommended subsections:

#### 13.1 Shared latency-minimization principle

#### 13.2 Structural difference between uplink and downlink

#### 13.3 Role of learning in the two links

### 14. Assumptions, scope boundaries, and theoretical limitations

This section should capture what the formulations assume.

Include:

- finite-blocklength approximation assumptions
- channel knowledge assumptions
- block-structure assumptions
- scheduling assumptions
- power-budget assumptions
- separations made for tractability

This section should stay methodological, not empirical.

### 15. Conclusion

The conclusion should summarize:

- the unified research problem
- the uplink methodological framework
- the downlink methodological framework
- the conceptual relationship between the proposed methods
- the fact that empirical validation is presented in a separate results report

## Appendices

### Appendix A. Complete notation table

- all symbols
- dimensions where useful
- uplink-only and downlink-only notation

### Appendix B. Detailed derivations

Include derivations that are too long for the main body:

- finite-blocklength expressions
- constrained reformulations
- intermediate optimization steps

### Appendix C. Algorithm summaries

Provide formal algorithm boxes or pseudocode for:

- uplink dynamic sub-block construction
- uplink constrained block solve
- downlink causal block scheduler
- learned-precoder integration into the scheduling loop

### Appendix D. Modeling assumptions by method

This appendix can list, for each method family:

- decision variables
- objective form
- constraint form
- online versus offline role

### Appendix E. Scenario definitions

This appendix should formally define:

- payload-completion scenario
- fixed-block-target scenario

## Non-negotiable coverage checklist

Do not finalize the main report until each item below is covered.

- motivation for finite-blocklength latency minimization
- shared notation
- shared latency definition
- shared constraint structure
- payload-completion scenario
- fixed-block-target scenario
- uplink system model
- uplink optimization problem
- uplink dynamic sub-block idea
- uplink online convergence method
- uplink learned Monte Carlo method
- downlink system model
- downlink interference-coupled formulation
- downlink optimization problem
- downlink online convergence method
- downlink bit-allocation logic after convergence
- downlink learned Monte Carlo method
- downlink architectural variants
- cross-link methodological synthesis
- assumptions and scope boundaries

## Suggested writing order

1. Introduction
2. Unified notation and common problem setting
3. Common optimization viewpoint
4. Uplink system model and optimization problem
5. Proposed uplink methodologies
6. Downlink system model and optimization problem
7. Proposed downlink methodologies
8. Cross-link methodological synthesis
9. Assumptions, scope boundaries, and theoretical limitations
10. Conclusion
11. Appendices
