int DIMENSIONS = 4;
int ACTIONS = 2;
int[] BUCKETS = {10,40,40,20};
double EPS = 0.0000000001; //1E-09

class Learner{
	// distance to obstacle
	// height above lower pipe
	// height below upper pipe
	// velocity
	// action

	float[][] minmax = {{-30,80},	//best guess
						// {-25,25},
						{50,100},
						{-140,200},
						{-4,8}};
	float epsilon, deltaEpsilon;
	float learningRate;
	float discount;

  int replayLength = 100;
	long replays = 0;

	float[] stateCoords;
	float[] statePrimeCoords;

	DuelingNet Q, Target;

	ArrayList<Experience> memory = new ArrayList<Experience>();
	int MAX_MEMORY = 1000000;
	int MIN_MEMORY = 40000;

	int copyFrequency = 1500;
	int segmentFrequency = 1000;

	int[] memorySegments = new int[replayLength];
	float priority = 0.65;
	float priorityCorrection = 0.1;
	float priorityCorrectionDelta = 0.000003;
	float probabilitySum = 0;

	int lastAction = 0;

	Learner(float deltaEpsilon, float learningRate, float discount){
		epsilon = 1;
		this.deltaEpsilon = deltaEpsilon;
		this.learningRate = learningRate;
		this.discount = discount;

		int[] actionLayers = {DIMENSIONS, 32, 2};
		int[] valueLayers = {DIMENSIONS, 32, 1};
		
		NeuralNet V = new NeuralNet(valueLayers, replayLength, true);
		NeuralNet VTarget = new NeuralNet(valueLayers, replayLength, true);
		NeuralNet A = new NeuralNet(actionLayers, replayLength, true);
		NeuralNet ATarget = new NeuralNet(actionLayers, replayLength, true);

		Q = new DuelingNet(V,A);
		Target = new DuelingNet(VTarget,ATarget);
	}

	void viewWorld(State s){
		if(statePrimeCoords == null){
			statePrimeCoords = stateCoords = normalise(s);	// initialise world
		}else{
			statePrimeCoords = normalise(s); 				// s' is view of the world
		}
	}

	Action act(){
		int choice;
		if(random(1) > epsilon){
			//act based on stateCoords
			double[] actions = getOutputOf(Q, stateCoords);
			choice = (actions[0] > actions[1]) ? 0 : 1;
		}else{
			choice = (random(1) > 0.5f) ? 0 : 1;
		}
		
		lastAction = choice;

		if(choice == 0){
			return Action.DOWN;
		}else{
			return Action.UP;
		}
	}

	void learn(float reward, boolean terminal){
		double[] s0, s1;
		s0 = new double[DIMENSIONS]; 
		s1 = new double[DIMENSIONS];
		for(int i = 0; i < DIMENSIONS; i++){
			s0[i] = stateCoords[i];
			s1[i] = statePrimeCoords[i];
		}
		addExperience(s0, s1, lastAction, reward, terminal);
		if(game.episodes % 2 == 0) experienceReplay();
		// prepare to act based on new state
		stateCoords = statePrimeCoords;
		// lower exploration rate
		if(memory.size() >= MIN_MEMORY) epsilon = max(0, epsilon - deltaEpsilon);	
	}

	void experienceReplay(){
		if(memory.size() < MIN_MEMORY) return;
		if(replays%copyFrequency == 0){
			Target.copy(Q);
		}
		if(replays%segmentFrequency == 0) calculateSegments();

		int[] replayId = new int[replayLength];
		Experience[] replay = new Experience[replayLength];
		double[][] s0Matrix = new double[replayLength][DIMENSIONS];
		double[][] s1Matrix = new double[replayLength][DIMENSIONS];
		
		for(int i = 0; i < replayLength; i++){
			replayId[i] = sampleSegment(i);
			Experience e = memory.get(replayId[i]);
			s0Matrix[i] = e.s0;	
			s1Matrix[i] = e.s1;	
		}

		double[][] maxActionMatrix;
		double[][] targetMatrix;

		Q.input(s1Matrix);
		Q.feedForward();
		maxActionMatrix = Q.output();

		if(memory.size() < MIN_MEMORY + 2000){
			targetMatrix = maxActionMatrix;
		}else{
			Target.input(s1Matrix);
			Target.feedForward();
			targetMatrix = Target.output();
		}
		
		Q.input(s0Matrix);
		Q.nesterov();
		Q.feedForward();

		double[][] outputMatrix = Q.output();

		for(int i = 0; i < replayLength; i++){
			Experience e = memory.get(replayId[i]);
			
			// select the maximising action
			int maximisingAction = 0;
			if(maxActionMatrix[i][1] > maxActionMatrix[i][0]) maximisingAction = 1;

			double maxFutureReward = targetMatrix[i][maximisingAction];

			if(e.terminal){
				targetMatrix[i][e.action] = (double) e.reward; 
			}else{
				targetMatrix[i][e.action] = (double) e.reward + discount * maxFutureReward; 
			}
			
			targetMatrix[i][1-e.action] = outputMatrix[i][1-e.action];
		}

		Q.target(targetMatrix);
		
		// bias annealing
		double[][] correction = new double[2][replayLength];
		double maxWeight = 0;
		for(int i = 0; i < replayLength; i++){
			correction[0][i] = correction[1][i] = pow(1f/memory.size() * 1f/probabilityOf(replayId[i]+1),priorityCorrection);
			maxWeight = max((float)maxWeight, (float)correction[0][i]);
		}
		SimpleMatrix correctionMatrix = new SimpleMatrix(correction).scale(1f/maxWeight);
		
		double error = Q.calculateError(correctionMatrix);
		averageError = averageError * replays / (double)(replays + 1) + error / (double)(replays + 1);
		double[] experienceErrors = Q.errorArray();
		
		for(int i = 0; i < replayLength; i++){
			double previousError = memory.get(replayId[i]).error;
			memory.get(replayId[i]).setError(experienceErrors[i]);
			
			if(previousError > experienceErrors[i]){
				bubbleDown(replayId[i]);
			}else{
				bubbleUp(replayId[i]);
			}
		}

		priorityCorrection = min(1,priorityCorrection+priorityCorrectionDelta);
		Q.backPropagate();
		replays++;
	}

	void addExperience(double[] s0, double[] s1, int action, float reward, boolean terminal){
		Experience e = new Experience(s0,s1,action,reward,terminal);
		if(memory.size() > MAX_MEMORY) memory.remove(0);
		memory.add(e);
		bubbleUp(memory.size()-1); 
	}

	private int sampleSegment(int seg){
		int s = (seg == 0) ? 0 : memorySegments[seg - 1];
		int e = (seg == replayLength - 1) ? memory.size() : memorySegments[seg];
		return min(memory.size() -1, floor(random(s,e))); // #TODO verify behaviour of random(x,y)
	}

	private void calculateSegments(){
		probabilitySum = 0;
		for(int i = 1; i <= memory.size(); i++) probabilitySum += pow(1f/float(i), priority);
		
		int segments = 0;
		float cumulative = 0;
		for(int i = 1; i <= memory.size(); i++){ // be careful to start with i = 1
			cumulative += probabilityOf(i);
			if(cumulative > (segments + 1) / (float) replayLength){
				memorySegments[segments] = i; // memorySegments[i] stores first index of segment(i+1)
				segments++;
			}
		}
	}

	private float probabilityOf(int i){
		return pow(1f/float(i), priority) / probabilitySum;
	}

	private double[] getOutputOf(DuelingNet N, float[] inputs){
		double[][] inputMatrix = new double[replayLength][DIMENSIONS];
		for(int i = 0; i < DIMENSIONS; i++) inputMatrix[0][i] = inputs[i];	
		N.input(inputMatrix);
		N.feedForward();
		return N.output()[0];
	}

	private float[] normalise(State s){
		float[] coordinates = new float[DIMENSIONS];
		for(int i = 0; i < DIMENSIONS; i++){
			float coord = s.getDimension(i);
			
			// update minmax if exceeded
			if(coord < minmax[i][0]) minmax[i][0] = coord;
			if(coord > minmax[i][1]) minmax[i][1] = coord;
			// normalise coord
			coord = (coord - minmax[i][0]) / (minmax[i][1] - minmax[i][0]);
			coordinates[i] = coord;
		}
		return coordinates;
	}

	private void bubbleUp(int i){
		while(i > 0 && memory.get((i-1)/2).error < memory.get(i).error){
			Collections.swap(memory, (i-1)/2, i);
			i = (i-1)/2;
		}
	}

	private void bubbleDown(int i){
		while((i+1)*2 < memory.size()){
			int maxChild = i*2+1;
			if(maxChild+1 < memory.size() && memory.get(maxChild).error < memory.get(maxChild+1).error) maxChild = maxChild+1;
			if(memory.get(i).error < memory.get(maxChild).error){
				Collections.swap(memory,i,maxChild);
				i = maxChild;
			}else{
				break;
			}
		}
	}
}

class DuelingNet {
	NeuralNet V;
	NeuralNet A;
	double[][] outputs;
	double[][] target;
	double[] error;

	DuelingNet(NeuralNet V, NeuralNet A){
		this.V = V;
		this.A = A;
	}

	void input(double[][] inputs){
		V.input(inputs);
		A.input(inputs);
	}

	double[][] output(){
		double[][] value = V.output();
		double[][] advantage = A.output();
		outputs = calculateQ(value, advantage);
		return outputs;
	}

	void feedForward(){
		V.feedForward();
		A.feedForward();
	}

	void nesterov(){
		V.nesterov();
		A.nesterov();
	}

	void backPropagate(){
		V.backPropagate();
		A.backPropagate();
	}

	void target(double[][] targetMatrix){
		target = targetMatrix;
	}

	double calculateError(SimpleMatrix correctionMatrix){
		SimpleMatrix dCost = new SimpleMatrix(target).minus(new SimpleMatrix(outputs)).negative();
		SimpleMatrix delta = dCost.transpose().elementMult(correctionMatrix);
		
		SimpleMatrix VDelta = new SimpleMatrix(1, delta.numCols());
		error = new double[delta.numCols()];
		for(int i = 0; i < delta.numCols(); i++){
			error[i] = delta.extractVector(false,i).elementSum(); // provide error array for prioritized EP
			VDelta.set(0,i, error[i]);
		}

		SimpleMatrix ADelta = delta.copy();
		
		for(int i = 0; i < ADelta.numCols(); i++){ // for each training example
			double dE = ADelta.extractVector(false, i).elementSum();
			for(int j = 0; j < ADelta.numRows(); j++){ // for each action
				if(ADelta.get(j,i) < EPS){ // == 0
					ADelta.set(j,i,(-1f/ACTIONS) * dE);
				}else{
					ADelta.set(j,i, (1f-(1f/ACTIONS)) * dE);
				}
			}
		}
		
		V.delta[V.ctLayers-1] = VDelta;
		A.delta[A.ctLayers-1] = ADelta;

		return dCost.elementPower(2).scale(0.5).elementSum();
	}

	double[] errorArray(){
		return error;
	}

	void copy(DuelingNet N){
		V.copy(N.V);
		A.copy(N.A);
	}

	private double[][] calculateQ(double[][] value, double[][] advantage){
		double[][] Q = new double[value.length][ACTIONS];
		// Q = V + A - 1/|A|(Ai)
		for(int i = 0; i < value.length; i++){
			double mean = 0;
			for(int j = 0; j < ACTIONS; j++) mean += advantage[i][j];
			mean /= advantage.length;
			for(int j = 0; j < ACTIONS; j++){
				Q[i][j] = value[i][0];
				Q[i][j] += advantage[i][j];
				Q[i][j] += mean;
			}
		}
		return Q;
	}
}

class Experience {
  double[] s0, s1;
  int action;
  float reward;
  boolean terminal;
  
  double error;

  Experience(double[] s0, double[] s1, int action, float reward, boolean terminal){
  	this.s0 = s0;
  	this.s1 = s1;
  	this.action = action;
  	this.reward = reward;
  	this.terminal = terminal;
  	this.error = 100;
  }

  void setError(double e){
  	this.error = e;
  }

  @Override
  public String toString(){
    return this.error + "";
  }
}