I did a bunch of paper-hopping starting from Diffusion model papers. I realized that there is so much we don't really understand about advances in RL, beyond this class. There are so many papers that deal with things that all build on each other. I feel like we may be in way over our heads if we try and do something actually on the cutting-edge, like diffusion models. There's just so much the exisiting theory is being built off of that we don't understand yet. 

So on the bright side: I think I found a really good paper that's at a level we can fully understand. 

Double-Q Learning: https://proceedings.neurips.cc/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf

The main good thing about this paper is: there is a whole section on "Future work" which outlines things that I think we can do and would make a great project. There's a whole set of algorithms that build off Q-learning that are talked about in this section, and there are concrete project ideas in this section. 

I am very much considering this to be a good project idea. Go through the paper and let me know your thoughts. 

-----------------------------------------

Further thoughts:
Basically the way the idea of Q learning advanced in this direction seems to be:

1. Deep Q Networks (DQN)
2. Double Deep Q learning
3. Deep Reinforcement Learning with Double Q-learning
4. Clipped Double Q-learning

We can talk in our paper about how these advanced gradually occurred. Then we can try and make some improvements based on the original Double-DQN paper and see where that gets us. Then we can look into the Clipped DQN paper and see if our ideas help with that too. 

We can come up with new experiments to try and test our ideas. In the original Double-DQN paper, they have an example of a roulette and some probabilistic estimator. We can come up with new ideas and environments to see how our ideas fare. 

----------------------------------------

- **Hypothesis** (Succinctly state your hypothesis)


- **Experiments** (What are some concrete experiments to test the hypothesis?)


- **Relevant Literature (**What papers or prior work is relevant?)


- **Why does it matter if we succeed? (**Why should we solve this problem.)

Ideas: "Additionally, the fact that we can construct positively biased and negatively
biased off-policy algorithms raises the question whether it is also possible to construct an unbiased off-policy reinforcement-learning algorithm, without the high variance of unbiased on-policy Monte-Carlo methods."

------------------------------------------

Resources:

- Clipped Double Q-learning: https://arxiv.org/pdf/1802.09477.pdf
- Deep Reinforcement Learning with Double Q-learning: https://arxiv.org/pdf/1509.06461.pdf

More stuff on Q-learning:
- Vanilla Deep Q Networks: https://towardsdatascience.com/dqn-part-1-vanilla-deep-q-networks-6eb4a00febfb
- Double Deep Q Networks: https://towardsdatascience.com/double-deep-q-networks-905dd8325412

---------------------------------------

BTW: the chain of papers was 

https://arxiv.org/pdf/2205.09991.pdf (Planning with Diffusion for Flexible Behavior Synthesis)

https://arxiv.org/pdf/2110.06169.pdf (OFFLINE REINFORCEMENT LEARNING
WITH IMPLICIT Q-LEARNING) 

https://arxiv.org/pdf/1802.09477.pdf (Addressing Function Approximation Error in Actor-Critic Methods)


