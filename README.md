## Deep Q-Learning From Scratch

#### Tabular Q-Learning, Deep Q-Learning, and DeepMind's research papers explained

### Introduction

*DISCLAIMER: This was written by a CS freshman about his experiences implementing the DeepMind research papers. I am not a machine learning expert. Some things may be subtly incorrect or completely wrong. (Please let me know if I've made such a mistake.)*

### Reinforcement Learning

### The Environment

The game we will try to learn will be a [Processing][processing] clone of Flappy Bird. 

It's an ideal toy environment for several reasons 

* Limited action space, *Jump* and *Don't Jump*
* Continuous reward feedback, $+1$ for each pipe passed, making the credit assignment problem less significant
* Limited world state, namely the incoming pipes, and the player's own state

The last point is important because we will **not** be implementing a key section of the DeepMind Atari work, namely using ConvNets (Convolutional Neural Networks) to learn features of the game to be fed into the Q-Network. Rather, we will engineer features to feed it, so that we can focus on how the Q-Learning section works.

To better gauge the relative strengths of the algorithms and see how the impacts of the various improvements, there will be different physics models, 

* **Move** - with no gravity
* **Jump** - a normal flappy bird clone

We'll also modify parameters to adjust the difficulty, e.g. ```gapHeight```, size of the gap between pipes.

### Tabular Q-Learning

Like neural networks, the Q-Learning algorithm has been around for a long time, since 1989. The *Q* in Q-Learning refers to a Q-function that specifies how good an action *a* is, in a given state *s*. Using an optimal Q-function, the agent can successfully navigate an environment to maximise its reward.

Q-Learning (sometimes stated as *SARSA*) is an algorithm for training towards this optimal Q-function. It boils down to a single equation for estimating Q-values based on new information:

$Q(s,a) = Q(s,a) + learningRate \times ( r + discount \times max[Q(s', a)]  - Q(s, a))$

* $s$ - the state the agent was originally in
* $a$ - the action the agent took
* $r$ - the reward the agent received
* $s'$ - the new state the agent is in

Let's break that down.

To calculate the new expected $Q(s, a)$ we find $r + discount \times max[Q(s', a)]$. That is, we not only consider the existing reward $r$, but also the maximum future reward we expect to get. The key insight here is that we can use our existing Q-function to find that in a form of dynamic programming. We iterate over all the possible actions for $s'$ and find the maximum reward we can get, $max[Q(s', a)]. We$ then multiply that by some $discount$ factor. This reflects the idea that future rewards are less valuable because they're uncertain, and also prevents the Q-values from exploding to infinity as long as $discount \lt 1$.

We take the new expected $Q(s,a)$ and then update the existing value using some $learningRate$. To understand this, look at the spectrum where

$learningRate = 0$

$ Q(s,a) = Q(s,a)$

$learningRate = 1$

$Q(s,a) = Q(s,a) + ( r + discount \times  max[Q(s', a)]  - Q(s, a))  = r + discount \times max[Q(s', a)]$

When $learningRate = 1$, we completely replace the old value with the new one. While this seems like a reasonable idea, it actually introduces a lot of instability into the system, especially if the environment is noisy. The new $Q(s,a)$ may be an outlier caused by an extremely favourable or unfavourable environment. Hence, we use e.g. a small $learningRate = 0.01$, to aggregate information over time.

However, we cannot always just pick $$max_a[Q(s,a)]$$, because (at least initially) the policy may not be perfect. This is the *exploitation-exploration* tradeoff. Sometimes we want to explore the world and try a non-optimal action if it may lead to a better outcome later on. Sometimes we want to exploit the policy we already know to try to maximise reward.

To do this we introduce an *$\epsilon$-greedy* approach, where $0\leq\epsilon\leq1$. Every iteration, we choose to explore with probability $\epsilon$ , and choose from the actions with uniform probability. The remaining $(1-\epsilon)$ of the time, we follow $Q(s,a)$. Over time we lower $\epsilon$ by some factor until $\epsilon = 0$.

Now that we (hopefully) understand how Q-Learning works, we can move on to the implementation

```java
// in the initialisation, we setup the agent to get the current state 
agent = new Learner(0.0000001f, 0.1, 0.95); // deltaEpsilon, learningRate & discount
agent.viewWorld(game.currentState());

// then in every iteration
for(int i = 0; i< frameSpeed; i++){
  game.takeAction(agent.act());         // agent takes an action 
  game.nextFrame();                     // the world updates from s to s'
  agent.viewWorld(game.currentState()); // the agent views s'
  int reward = game.points;				// and gets reward r
  if(game.terminal){
    // If the game has ended, we apply a penalty corresponding to the maximum score
    // obtained so far. In effect, as the agent improves the penalties get worse
    reward = min(-1, -1 * game.highScore);
    record(game.points); 
  }
  agent.learn(reward); // given (s, a, r, s') the agent can now learn
}
```

The ```Learner``` class looks like this

```java
int ACTIONS = 2; // number of actions
int DIMENSIONS = 4; // number of dimensions of inputs
// Here, the dimensions are 
// Distance to next pipe, Distance to upper pipe, Distance to lower pipe, Agent's velocity

int[] BUCKETS = {30,40,40,20}; // we discretize the inputs into buckets

class Learner{
    // lookup table
	float[][][][][] Q = new float[BUCKETS[0]][BUCKETS[1]][BUCKETS[2]][BUCKETS[3]][ACTIONS];
	// minimum and maximum to normalise (feature scaling actually) inputs
    float[][] minmax = {{10,90}, {-25,25},{-25,25},{-7,7}};
  
	float epsilon, deltaEpsilon;
	float learningRate;
	float discount;

	int[] stateCoords;      // s
	int[] statePrimeCoords; // s'
	int lastAction = 0;     // a

	// constructor omitted
}
```
The ```snap``` function discretizes the continuous values of the state. Increasing or decreasing the number of buckets for each dimension affects the effective resolution for each dimension.
```java
private int[] snap(State s){
	int[] coordinates = new int[DIMENSIONS];
	for(int i = 0; i < DIMENSIONS; i++){
		float coord = s.getDimension(i);
		// update minmax if exceeded
		if(coord < minmax[i][0]) minmax[i][0] = coord;
		if(coord > minmax[i][1]) minmax[i][1] = coord;
		// normalise coord by min-max scaling
		coord = (coord - minmax[i][0]) / (minmax[i][1] - minmax[i][0]);
		coordinates[i] = round(coord * ((float) BUCKETS[i] - 1));
	}
	return coordinates;
}
```
The ```viewWorld``` function simply updates the agent's ```statePrimeCoords```. 
```java
void viewWorld(State s){
  if(statePrimeCoords == null) stateCoords = snap(s); // initialise agent
  statePrimeCoords = snap(s); 
}
```
The ```act``` function uses an $\epsilon$-greedy approach to pick actions.
```java
Action act(){
  int choice;
  if(random(1) > epsilon){ 	// act based on state
    float[] actions = Q[stateCoords[0]][stateCoords[1]][stateCoords[2]][stateCoords[3]];
    choice = (actions[0] > actions[1]) ? 0 : 1;
  }else{					// pick an action at random
    choice = (random(1) > 0.5f) ? 0 : 1;
  }
  lastAction = choice; 		// store the action for later
 
  if(choice == 0){
    return Action.DOWN;
  }else{
    return Action.UP;
  }
}
```
Now we get to the meat of the algorithm, the ```learn``` function, which essentially implements the value update equation mentioned above.
```java
void learn(float reward){
  // Q(s,a)
  float previousQ = Q[stateCoords[0]][stateCoords[1]][stateCoords[2]][stateCoords[3]][lastAction];
  // Q(s'a) for all a
  float[] possibleActions = Q[statePrimeCoords[0]][statePrimeCoords[1]][statePrimeCoords[2]][statePrimeCoords[3]];
  // max Q(s'a)
  float maxFutureReward = max(possibleActions[0], possibleActions[1]);

  //Q update equation
  Q[stateCoords[0]][stateCoords[1]][stateCoords[2]][stateCoords[3]][lastAction] = previousQ + learningRate * (reward + discount * maxFutureReward - previousQ);

  // prepare to act based on new state
  stateCoords = statePrimeCoords;

  // lower exploration rate
  epsilon = max(0, epsilon - deltaEpsilon);	
}
```

And that's it! The full code for this stage is available [here][stage_2]. You should be able to get fairly impressive results in under a few minutes of training (press ```UP``` to increase the frame speed), and the agent should play nearly-perfectly.

### Deep Q-Learning

#### Neural Nets as a Q-function approximator

In the tabular version we used a lookup table as the Q function. But there's some significant downsides to this solution.

*     There's limited generalisability. That is if we have a new state $y$ that differs from seen state $x$ by some small distance e.g. 1 indice, the agent doesn't learn from $x$ at all. That is, it doesn't generalize well to previously unseen states.

*     The limiting factor is memory. A lookup table for a state like the Atari 2600 pixel frame would be impossibly large to store.

With Deep Q-Learning, we replace the lookup table for the Q-function with a neural net to approximate the Q-function. Specifically, we have a neural net which takes in the inputs which describe the state (note that we only describe the state, and not the action), and have it produce an array of outputs describing the Q-values for all the actions. This saves us from performing a forward pass for each action, as compared to a system where we describe the action and the state as inputs and produce a single Q-value as the output.

However, since we only ever update one Q-value, $Q(s,a)$, we can only correct one of the outputs of the neural net during backpropagation. So, when we set the target values for the neural net, we update one and leave the other the same. Hence only one output node 'learns' because the other output nodes have $error = 0$.

Specifically, we 


1.                                 Input s and feed forward, to produce an array $Q(s,a)$
2.                                 Choose the best action $a*$ based on $max Q(s,a)$
3.                                 Input $s'$ and feed forward to produce an $array Q(s',a)$
4.                                 Find the maximum reward from $Q(s',a)$
5.                                 Update the entry for $a*$ in $Q(s, a)$ and use that as the target to backpropagate

#### Experience Replay

If we were to directly train the system like this, using stochastic gradient descent, we would find the system to be extremely unstable and with a tendency to diverge (i.e. not find any solution). The reason for this is *catastrophic forgetting*. 

For stochastic gradient descent to work, there is an assumption that the underlying distribution of training examples is roughly the same over time. Obviously, with online learning that assumption is definitely false, as we introduce new training examples into the training set over time. However, that can push the neural net towards one extreme to learn one behaviour, and saturate the training set with examples that result from that new behaviour. The neural net then rises to the occasion to solve the new examples, while forgetting the old examples that led to the first behaviour. 

The proposed solution to this is **experience replay** with mini-batch gradient descent. We store every experience, where an experience consists of $(s, a, r, s')$ in a memory. That is, instead of learning from the new experience every time, we insert the new experience into the memory. Then, we replay a subset, or mini-batch, of these examples to train the neural net. This ensures that both old and new experiences are trained on, and helps to even out the training distribution over time.

#### Target Net

However, as the targets are generated using the same net, there is a high likelihood that after updating $Q(s,a)$, $Q(s',a)$ will shift in the same direction. An increase in $Q(s,a)$ would lead to an increase in $max Q(s',a)$, and amplify the effects of each update. This tends to lead to instability in the algorithm and oscillation in its behaviour. 

A key innovation from the Deep Q-Learning paper was the idea of having a separate net to generate target values and update it after some delay. That is, instead of using the main net to calculate the future rewards, we use the target net, and then copy the weights from the main net to the target net every $T$ steps, where $T$ is some significant delay. This increases the stability of the algorithm as the targets in backpropagation do not shift as wildly as they might if using the same net.

Specifically, we 
1. Select a mini-batch of experiences
2. Input $s$ into $Q$ and feed forward to produce an array $Q(s,a)$
3. Choose the best action $a*$ based on $max Q(s,a)$
4. Input $s'$ into Target and feed forward to produce an $array Target(s',a)$
5. Find the maximum reward from $Target(s',a)$
6. Update the entry for $a*$ in $Q(s, a)$ and use that as the target to backpropagate
7. Copy Q to Target every $T$ steps

The toy environment that we are playing with is not really a good candidate for Deep Q-Learning, mainly because it's so trivial. However we'll press on and to look at how it can work regardless. The full starter code is [here][pre_stage_3] and includes an implementation of a regression neural net with Nesterov momentum and RMSProp.

```java
int replayLength = 100;
long replays = 0;

NeuralNet Q, Target;

int[] layers = {DIMENSIONS, 32, 32, ACTIONS}; // 2 hidden layers of 32 units
Q = new NeuralNet(layers, replayLength, true);
Target = new NeuralNet(layers, replayLength, true);
```

```java
// in act() we replace the table lookup
// float[] actions = Q[stateCoords[0]][stateCoords[1]][stateCoords[2]][stateCoords[3]];
// ... with
double[] actions = getOutputOf(Q, stateCoords);
// ...

private double[] getOutputOf(NeuralNet N, float[] inputs){
  // these lines transform the input into a minibatch matrix with 0s everywhere
  // except at i = 0, the inputs
  double[][] inputMatrix = new double[replayLength][DIMENSIONS];
  for(int i = 0; i < DIMENSIONS; i++) inputMatrix[0][i] = inputs[i];	
  
  N.input(inputMatrix);
  N.feedForward();
  return N.output()[0]; // and return only the outputs from the inputs
}
```

We store each transition as an ```Experience```, with $s,a,r,s'$ and a boolean indicating if it is a terminal state.

```java
ArrayList<Experience> memory = new ArrayList<Experience>();
int MAX_MEMORY = 1000000;
int MIN_MEMORY = 40000; // the agent collects MIN_MEMORY before starting experience replay
  
class Experience {
  double[] s0, s1;
  int action;
  float reward;
  boolean terminal;
  // constructor omitted
}
```

The learn function is dramatically simplified. The new experience is stored and ```experienceReplay``` is executed every 2 frames (to reduce computation).

```java
// in learn() we replace the Q update function with
// ...
addExperience(s0, s1, lastAction, reward, terminal);
if(game.episodes % 2 == 0) experienceReplay()
// ...

void addExperience(double[] s0, double[] s1, int action, float reward, boolean terminal){
  Experience e = new Experience(s0,s1,action,reward,terminal);
  if(memory.size() > MAX_MEMORY) memory.remove(0); // remove oldest
  memory.add(e);
}
```

 The bulk of the work is done in the ```experienceReplay``` function

```java
void experienceReplay(){
   // the agent first collects MIN_MEMORY experiences using a uniform exploration policy
  if(memory.size() < MIN_MEMORY) return;
  
  int[] replayId = new int[replayLength];
  double[][] s0Matrix = new double[replayLength][DIMENSIONS];
  double[][] s1Matrix = new double[replayLength][DIMENSIONS];

  for(int i = 0; i < replayLength; i++){
    replayId[i] = floor(random(memory.size())); 	// pick random experience
    Experience e = memory.get(replayId[i]);
    s0Matrix[i] = e.s0;								// store s0 into matrix	
    s1Matrix[i] = e.s1;								// and s1 into matrix
  }

  // find maxFutureReward using s1 matrix
  double[][] targetMatrix;
  if(memory.size() < MIN_MEMORY + TARGET_WAIT){ // collect an additional TARGET_WAIT
    Q.input(s1Matrix);							// experiences and replays before 
    Q.feedForward();							// using TargetNet, i.e. initialise 
    targetMatrix = Q.output();					// TargetNet not randomly, but with 													// weights trained over TARGET_WAIT
  }else{
    Target.input(s1Matrix);						// otherwise, feedForward s1
    Target.feedForward();
    targetMatrix = Target.output();				// and store the output
  }
  
  // feedforward s0 to obtain outputs
  // error between target and output will be used to backpropagate
  Q.input(s0Matrix);
  Q.nesterov(); // this applies nesterov momentum correction to the neural net
  Q.feedForward();	

  double[][] outputMatrix = Q.output();

  // at this point, the targetMatrix and outputMatrix have been filled
  
  for(int i = 0; i < replayLength; i++){
    Experience e = memory.get(replayId[i]);
    // find the maxFutureReward based on max(targetMatrix[i]) 
    double maxFutureReward = (targetMatrix[i][0] > targetMatrix[i][1])? targetMatrix[i][0] : targetMatrix[i][1];
    // calculate the actual targets
    if(e.terminal){ // if terminal, maxFutureReward = 0
      targetMatrix[i][e.action] = (double) e.reward; 
    }else{
      targetMatrix[i][e.action] = (double) e.reward + discount * maxFutureReward; 
    }
    // for the unselected action, set the target to what it output, 
    // i.e. the error for the unselected action should be 0
    targetMatrix[i][1-e.action] = outputMatrix[i][1-e.action];
  }

  Q.target(targetMatrix);
  double error = Q.calculateError();
  averageError = averageError * (double) replays / ((double) replays + 1) + error / ((double) replays + 1);
  Q.backPropagate(); // after calculating error, backpropagate

  // copy the weights from Q to Target every TARGET_WAIT replays
  if(replays % TARGET_WAIT == 0) Target.copy(Q);
  replays++;
```

It should take approximately 100,000 - 200,000 episodes for the agent to train. You'll quickly realise that this is *very, very* much slower compared to the tabular Q-learner. The tabular Q-learner makes each update and action choice with constant time $\Theta(1)$, whereas the DQL learner has to do experience replay (proportional to net size and ```REPLAY_LENGTH```), and evaluate the Q-net (proportional to net size). 

*Additionally, I couldn't get OpenCL working with processing, so the system isn't parallelised or taking advantage of any GPU. In a real system, (with a lot larger nets and more data), the matrix multiplications would be performed far more efficiently on an array of GPUs.*

To speed it up, we can alter several variables,

* Let the agent only act every $k$ frames, and ```fall()``` for the rest. This significantly boosts training speed. In the DQL paper, they propose a similar speedup, except the action is repeated, i.e. the same action is applied for all $k$ frames. In effect, this simulates some measure of human reaction time.
* Carry out ```experienceReplay()``` only every $n$ frames, because experience replay is costly. This allows the agent to collect more experiences overall in the same amount of time.

The completed code for this stage is [here][stage3].

However, even after a long period, the agent might not converge to a good policy, when compared with the tabular Q-learner. Time to bring in some additional help.

### Double Q-Learning and Double DQN

If we expand the Q-learning equation, the equation for calculating targets/updates looks like this:

$maxFutureReward$

$ = r + discount \times max[Q(s', a)] $

$= r + discount \times max[r + discount \times max[Q(s'', a)]]$

$= r + discount \times max[r + discount \times max[r+discount\times max[r+discount\times ...]]]$

The max operator tends to cause an overestimation of how good any given state is, as it accumulates any positive noise in the Q-value estimation. Over time, that accumulation can cause the system to become incredibly inaccurate, or diverge.

To improve upon this, an original innovation in Q-Learning, Double Q-Learning, was to have two lookup tables and randomly assign experiences to each lookup table. Then, instead of using the same lookup table for calculating updates, we use one two pick the best future action and one to calculate the value of that action.

$maxFutureReward = r + discount \times max[Q_1( s', max_a Q_2(s', a) )]$

We can apply this in Deep Q-Learning by using two nets. But, we already have two nets, the Q net and the Target net, hurrah! With Double DQN, we use the Q net to pick the best action, and the Target net to estimate the value of that action.

Thankfully, the modifications to achieve this are rather simple. We just need to alter the code that calculates the targets we use in backpropagation during `experience_replay()`

```java

double[][] maxActionMatrix;	// this matrix will store the output of Q
double[][] targetMatrix;	// this matrix will store the output of Target

Q.input(s1Matrix);			// feed in s1 into Q to determine maxAction
Q.feedForward();
maxActionMatrix = Q.output();

if(memory.size() < MIN_MEMORY + 2000){
  targetMatrix = maxActionMatrix;
}else{
  Target.input(s1Matrix);			// feed in s1 into Target to determine actual value
  Target.feedForward();
  targetMatrix = Target.output();
}

// ... lines omitted for brevity

for(int i = 0; i < replayLength; i++){
  Experience e = memory.get(replayId[i]);
  // select the maximising action
  int maximisingAction = 0;
  if(maxActionMatrix[i][1] > maxActionMatrix[i][0]) maximisingAction = 1;
  // maxFutureReward is now the value the maximisingAction in targetMatrix
  double maxFutureReward = targetMatrix[i][maximisingAction];

// for comparison, the old version looked like this
// double maxFutureReward = 
// (targetMatrix[i][0] > targetMatrix[i][1]) ? targetMatrix[i][0] : targetMatrix[i][1];
  
// ... lines omitted for brevity
```

The remainder of `experience_replay()` is exactly the same. The completed code for this stage is [here](undefined).

Empirically, I found that this agent converges to a better policy, and converges to it faster than the vanilla DQL agent. However, faster is relative and it's still pretty slow. We'll try to improve training speed in the next addition.

### Prioritized Experience Replay

With experience replay, we selected experiences using a uniform probability distribution, each experience was equally likely to be picked. However, it would be sensible to instead pick experiences from which we are likely to learn the most. That's the essence of prioritised experince replay.

The question then is, how do we find experiences which we can learn most from, without actually learning them and seeing the change? The suggested method is to use the Temporal Difference (TD) error as a proxy to learning where 

$TD\ Error = |Q(s,a) - (r + maxQ(s',a))|$

Intuitively, this is a measure of how 'surprising' any given transition is. For example, the transition from $0$ reward to $1$ reward when passing a pipe would have a high $TD\ Error$ and should be replayed more often. Similarly with the transition when hitting a pipe and a $-1$  penalty results.

#### Proportional Prioritization and Rank-based Prioritization

Now that we have a measure of how important any experience is, we might want to greedily replay only the set of experiences with the highest $TD\ Error$. But the obvious problem with that is that initially low $TD\ Error$ experiences may *never* be replayed. Additionally, it may cause the neural net to overfit and be weaker in the long run. (You might be noticing a trend here, in that we always have to be careful not to push the system one way or the other.) The proposed solution is a **stochastic prioritization**, with either 

* proportional prioritization, based on $TD\ Error_i$
* rank based prioritization, based on rank of $TD\ Error_i$ when sorted across the memory

$$P(i) = \frac{p_i^a}{\sum_{k} p_k^a}$$, if proportional : $$p_i = |TD\ Error_i|$$  if rank based: $$p_i = \frac{1}{rank(|TD\ Error_i|)}$$ 

$a$ is a priority factor, where $a=0$  is no prioritization, i.e. drawing from a uniform distribution. (When $a=0$, $p_i^a = 1$ for all $i$, hence the uniform distribution). By manipulating $a$ we can control how much the agent prioritizes experiences, and prevent the problem with overfitting to high $TD\ Error$ experiences.

Comparing between proportional and rank-based prioritization, theoretically rank-based prioritization should be better, because it isn't affected by outliers in $TD\ Error$. (Empirically, the paper found that there were differences, but they weren't significant.)  We'll be implementing rank-based prioritization.

#### Sorting experiences

It would be very computationally costly to keep sorting the list of experiences by constantly fluctuating $TD\ Error$ . The paper proposes an innovative compromise for rank-based prioritization by using a [binary max heap structure][binary heap wiki] in an unorthodox manner.

Examining a binary max heap, the underlying data structure (an ArrayList in our case), is *nearly-sorted*. That is, due to the nature of the heap structure, each sub-tree is a sorted heap. This means that each element $i$ is at larger than at least $k$ elements after it, where $k$ is the size of $i$'s subtree.

Well that was confusing. It's easier to see visually.

For a tree like this:



The underlying array is *nearly-sorted*

10, 8, 5, 6, 4, 3, 2, 2, 1

For our purposes this will do because we're only picking prioritized experiences and we don't need the guarantee of a perfectly sorted list. This allows us to insert experiences and modify $TD\ Error$ in $\Theta(logN)$ time, which is far better than sorting after every experience replay in $\Theta(NlogN)$ time.

*To do this, I implemented a `bubbleUp()` and `bubbleDown()` function to update the heap. The implementation of these functions are not shown.*

We modify `addExperience()` to perform this.

```java
void addExperience(double[] s0, double[] s1, int action, float reward, boolean terminal{
  Experience e = new Experience(s0,s1,action,reward,terminal);
  // instead of removing the 0th, i.e. the oldest memory, we remove the last,
  // the memory with the nearly-lowest TD-error
  if(memory.size() > MAX_MEMORY) memory.remove(memory.size()-1);
  memory.add(e);
  // bubbleUp swaps the memory with its parent in the tree until its parent has a larger TD error, maintaining the binary max heap
  bubbleUp(memory.size()-1); 
}
```

We also need to modify `experienceReplay()` to store the $TD\ Errors$ and update them.

```java
// in experienceReplay()
for(int i = 0; i < replayLength; i++){
  // in the experience replay loop
  // ...
  // TDError = |target - out|
  //		 = |(e.reward + discount * maxFutureReward) - out|
  double TDError = abs((float)(targetMatrix[i][e.action] - outputMatrix[i][e.action]));
 
  // update
  double previousError = memory.get(replayId[i]).error;
  memory.get(replayId[i]).setError(TDError); // set the new error
  if(previousError > TDError){
    bubbleDown(replayId[i]); // bubble down since TDError is now smaller
  }else{
    bubbleUp(replayId[i]);   // bubble up since TDError is now bigger
  }
}
```



#### Stratified Sampling

We'll further improve on this technique with [stratified sampling][stratified sampling wiki]. Stratified sampling involves selecting items across different strata of the population to obtain a representative sample. In our case the strata will be equal segments of $p_i = \frac{1}{rank(|TD\ Error_i|)}$, and we'll have the same number of strata as our `REPLAY_LENGTH`, so each experience in the replay will be in a different $TD\ Error$ range, allowing us to prevent catastrophic forgetting of low $TD\ Error$ experiences but also learning more often from high $TD\ Error$ experiences.

We'll implement this with a `calculateSegments()` function.

```java
private float probabilityOf(int i){
  // this lets us find P(i) = p_i/sum(p_k),
	return pow(1f/float(i), priority) / probabilitySum;
}

private void calculateSegments(){
  // update the value of probabilitySum, so that we correctly calculate P(i)
  probabilitySum = 0;
  for(int i = 1; i <= memory.size(); i++)
    probabilitySum += pow(1f/float(i), priority);

  int segments = 0;
  float cumulative = 0;
  for(int i = 1; i <= memory.size(); i++){ 	// be careful to start with i = 1
    cumulative += probabilityOf(i);	
    // 0 <= cumulative <= 1, because we add P(i), and not p_i
    
    // when the total so far is bigger than its segment,
    // i.e. if replayLength = 10, segments = 2, cumulative = 0.31,
    // then store the current experience as the 'break point' for this segment
    
    if(cumulative > (segments + 1) / (float) replayLength){
      // memorySegments[i] stores first index of segment(i+1)
      memorySegments[segments] = i; 
      segments++; // move on to next segment
    }
  }
}
```
We then uniformly sample from each segment to fill in each experience in the replay, and this approximates selecting each experience with $P(i)$.

```java
private int sampleSegment(int seg){
  // start at 0 if first segment 
  int s = (seg == 0) ? 0 : memorySegments[seg - 1];
  // end at last experience in memory if last segment
  int e = (seg == replayLength - 1) ? memory.size() : memorySegments[seg];
  // uniformly sample between s and e
  return min(memory.size() -1, floor(random(s,e))); 
}

// ... in experienceReplay() we alter the line 
// 	replayId[i] = floor(random(memory.size())); 
// to
replayId[i] = sampleSegment(i);

```

#### Bias Annealing

Well that was a long section. But we're not done yet! :(

When using a neural net, one of the assumptions we make is about distribution of training examples. Remember when we introduced experience replay, the purpose was to prevent catastrophic forgetting and to smoothen out the distribution. *But*, by prioritizing experiences, we've introduced a new source of bias into the system, that we have to correct for. 

Intuitively, we can think of this bias as affecting the gradient as we converge towards a solution. High $TD\ Error$ experiences are likely to result in larger gradients (more learning), and likely to be played more often. Frequent updates of large gradients create a lot of instability in the neural net, especially when it's converging towards a solution and needs to 'slow down'. Hence, we want to gradually anneal the bias (at the start the bias has little effect because the policy is still changing and there's a lot of noise in state and targets) as we are converging to the solution.

This is done with Importance-Sampling (IS) weights.

$$w_i = (\frac{1}{N}\times\frac{1}{P(i)})^\beta$$, $$\delta_i=w_i\delta_i$$

A new term $\beta$ is introduced, as the priority correction factor. We anneal $\beta$ from $0$ to $1$ such that at $1$ the bias is fully compensated for. Then we multiply the gradient $\delta_i$ by $w_i$ as compensation.

The equation above makes intuitive sense if we examine it. In an unprioritised system, we selected each experience with probability $\frac{1}{N}$. In the prioritised system we select each experience with probability $P(i)$. We divide by $P(i)$ and multiply by $\frac{1}{N}$ so that $w_i$ effectively 'undoes' the bias when it modifies the magnitude of the gradient. This way, high $TD\ Error$ experiences are still replayed many times, but the gradient of each is made small enough so that it can converge to a solution.

We continue to modify `experienceReplay()` to do this.

```java
// ... in experienceReplay()
double[][] correction = new double[2][replayLength];
double maxWeight = 0;
for(int i = 0; i < replayLength; i++){
  // calculate w_i
  correction[0][i] = correction[1][i] = 
    pow(1f/memory.size() * 1f/probabilityOf(replayId[i]+1),priorityCorrection);
  // find max(w_i)
  maxWeight = max((float)maxWeight, (float)correction[0][i]);
}
// create a correctionMatrix of w_i, normalised by 1/max(w_i)
// normalisation prevents delta from exploding, as it guarantees the correction
// will be <= 1 and increases the overall stability.
SimpleMatrix correctionMatrix = new SimpleMatrix(correction).scale(1f/maxWeight);
// multiply delta_i by the correctionMatrix
Q.delta[Q.ctLayers-1] = Q.delta[Q.ctLayers-1].elementMult(correctionMatrix);
```
