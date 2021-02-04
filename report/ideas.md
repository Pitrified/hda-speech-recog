# Ideas for the report

## Introduction

##### Tips

Paper contribution (what you do in the paper), problem, approach (technique
used + novelty), value ().

Paper structure (this report is structured as follows bla bla).

## Related

## Model

### Processing pipeline

##### Tips

High level description of the approach: which processing blocks you used, what
they do (in words) and how these were combined

High-level description of the involved processing blocks, i.e., you describe
the {\it processing flow} and the rationale behind it

Nice diagram

Describe the problem we are solving, task, number of words...

### Signals and Features

##### Add

Valutazione della scelta di fare preprocessing separato.
Offline/online augmentation

##### Tips

### Learning Framework

##### Add

CyclicLR has a paper describing it: https://arxiv.org/abs/1506.01186

##### Tips

Describe the learning strategy, the learning model, its parameters, any
optimization over a given parameter set

Diagram with details of learning framework

## Results

##### Add

Plots to describe Fscore as function of the learning parameters.

Intro su come leggere i grafici, gruppi di gruppi di colonne.

##### Tips

Progressive and logical manner, starting with simple things and adding details.
Address one concept at a time.

Hyper-parameters, show selected results for several values of these. Tables are
a good approach to concisely visualize the performance as hyper-parameters
change. How architectural choices affect the overall performance.

## Conclusions

##### Add

Check results consistency: cat\_acc high, fscore low: bad inputs somewhere.
Also had same problem in training.

Writing report a lot slower than I thought.

##### Tips

What I would like to see here is:
* a very short summary of what done, 
* some (possibly) intelligent observations on the relevance and applicability
  of your algorithms / findings, 
* what is still missing, and can be added in the future to extend your work.

The idea is that this section should be useful and not just a repetition of the
abstract (just re-phrased and written using a different tense...).

Moreover: being a project report, I would also like to see a specific paragraph
stating 
* what you have learned
* any difficulties you may have encountered
