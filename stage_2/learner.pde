int DIMENSIONS = 4;
int[] BUCKETS = {30,40,40,20};

class Learner{
	// distance to obstacle
	// height above lower pipe
	// height below upper pipe
	// velocity
	// action 


	float[][][][][] Q = new float[BUCKETS[0]][BUCKETS[1]][BUCKETS[2]][BUCKETS[3]][2];
	float[][] minmax = {{10,90},	//best guess
						{-25,25},
						{-25,25},
						{-7,7}};
	float epsilon, deltaEpsilon;
	float learningRate;
	float discount;

	int[] stateCoords;
	int[] statePrimeCoords;

	int lastAction = 0;

	Learner(float deltaEpsilon, float learningRate, float discount){
		epsilon = 1;
		this.deltaEpsilon = deltaEpsilon;
		this.learningRate = learningRate;
		this.discount = discount;
	}

	void viewWorld(State s){
		if(statePrimeCoords == null){
			statePrimeCoords = stateCoords = snap(s);	// initialise world
		}else{
			statePrimeCoords = snap(s); 				// s' is view of the world
		}
	}

	Action act(){
		int choice;
		if(random(1) > epsilon){
			//act based on stateCoords
			float[] actions = Q[stateCoords[0]][stateCoords[1]][stateCoords[2]][stateCoords[3]];
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

	void learn(float reward){
		float previousQ = Q[stateCoords[0]][stateCoords[1]][stateCoords[2]][stateCoords[3]][lastAction];
		float[] possibleActions = Q[statePrimeCoords[0]][statePrimeCoords[1]][statePrimeCoords[2]][statePrimeCoords[3]];
		float maxFutureReward = max(possibleActions[0], possibleActions[1]);

		//Q update equation
		Q[stateCoords[0]][stateCoords[1]][stateCoords[2]][stateCoords[3]][lastAction] = previousQ + learningRate * (reward + discount * maxFutureReward - previousQ);
		
		// prepare to act based on new state
		stateCoords = statePrimeCoords;

		// lower exploration rate
		epsilon = max(0, epsilon - deltaEpsilon);	
	}

	private int[] snap(State s){
		int[] coordinates = new int[DIMENSIONS];
		for(int i = 0; i < DIMENSIONS; i++){
			float coord = s.getDimension(i);
			
			// update minmax if exceeded
			if(coord < minmax[i][0]) minmax[i][0] = coord;
			if(coord > minmax[i][1]) minmax[i][1] = coord;
			
			// normalise coord
			coord = (coord - minmax[i][0]) / (minmax[i][1] - minmax[i][0]);
			coordinates[i] = round(coord * ((float) BUCKETS[i] - 1));
		}
		return coordinates;
	}
}