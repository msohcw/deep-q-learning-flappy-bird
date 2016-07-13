enum PhysicsModel{
	MOVE, JUMP, HOVER
}

enum Action{
	UP, DOWN
}

class FlappyBird {
	PhysicsModel world;
	ArrayList<Obstacle> obstacles = new ArrayList<Obstacle>();

	int obstacleGap = 100;
  	int obstacleWidth = 20;
	int obstacleSpeed = 2;
  	
  	int difficulty = 50;
  	int playerWidth = 10;
  	int playerHeight = 10;

  	PVector position, velocity, acceleration;

	FlappyBird(PhysicsModel w){
		world = w;
		
		reset();
	}

	public void nextFrame(){
		
		// player
		applyForces();
		velocity.y += acceleration.y;
		position.y += velocity.y;
		
		// obstacles
		for (int i = 0; i < obstacles.size(); i++) {
		  Obstacle o = obstacles.get(i);
		  o.advance(obstacleSpeed);
		  if(o.passed()) obstacles.remove(i);
		}


		Obstacle last = obstacles.get(obstacles.size()-1);
		if(last.topLeft.x + obstacleWidth + obstacleGap <= stageWidth) addObstacleAt(stageWidth);
		
		// collisions

		Obstacle first = obstacles.get(0);

		if(position.y + playerWidth/2 > stageHeight) reset();
		if(position.y - playerWidth/2 < 0) reset();
		//upper
		if(intersectAABB(position.x,position.y-playerHeight/2, playerWidth, playerHeight,first.topLeft.x,0,obstacleWidth,first.topLeft.y-first.gapHeight)) reset();
		//lower
		if(intersectAABB(position.x,position.y-playerHeight/2, playerWidth, playerHeight,first.topLeft.x,first.topLeft.y,obstacleWidth,stageHeight-first.topLeft.y)) reset();
			
	}

	public void takeAction(Action a){
		switch(a){
			case UP:
				jump();
				break;
			case DOWN:
				fall();
				break;
		}
	}

	public State currentState(){
		return new State();
	}

	public void draw(){
		noStroke();
		fill(#CC0000);
		rect(position.x, position.y-playerHeight/2, playerWidth, playerHeight);

		fill(#00CC00);
		stroke(#00AA00);
		for (int i = 0; i < obstacles.size(); i++) {
		  Obstacle o = obstacles.get(i);
		  
		  //upper
		  rect(o.topLeft.x, 0, obstacleWidth, o.topLeft.y - o.gapHeight);
		  //lower
		  rect(o.topLeft.x, o.topLeft.y, obstacleWidth, stageHeight - o.topLeft.y);
		}

		Obstacle first = obstacles.get(0);
		stroke(#FF0000);	
		rect(first.topLeft.x,first.topLeft.y,obstacleWidth,stageHeight-first.topLeft.y);
		rect(first.topLeft.x,0,obstacleWidth,first.topLeft.y-first.gapHeight);
	}

	// internal game methods

	private void reset(){
		obstacles.clear();

		position = new PVector(0, stageHeight/2);
		velocity = new PVector(0,0);
		acceleration = new PVector(0,0);

		for(int i = 0; i < floor(stageWidth / (obstacleGap + obstacleWidth)); i++){
			addObstacleAt(obstacleGap + (obstacleGap + obstacleWidth) * i);	
		}
	}

	private void addObstacleAt(int x){
		PVector gapCenter = new PVector(x, position.y + random(-difficulty/2, difficulty/2));
		int gapHeight = difficulty + round(random(difficulty));

		PVector topLeft = gapCenter.copy();
		topLeft.y += gapHeight/2;

		Obstacle o = new Obstacle(topLeft, gapHeight, obstacleWidth);
		obstacles.add(o);
	}

	private void applyForces(){
		switch(world){
			case MOVE:
				break;
			case JUMP:
				acceleration.y = 0.25;
			case HOVER:
				acceleration.y += 0.1;
		}
	}

	private void jump(){
		switch(world){
			case MOVE:
			case JUMP:
				// if(velocity.y > 0) velocity.y -= 7;
				velocity.y = -5; // flappy bird is not the real life
				break;
			case HOVER:
		}
	}

	private void fall(){
		switch(world){
			case MOVE:
			case JUMP:
			case HOVER:
		}
	}
}

class State{
	float[] coordinates;
	
	State(){
		coordinates = new float[DIMENSIONS];
	}

	State(float[] m){
		coordinates = m;
	}

	void setDimension(int i, float x){
		coordinates[i] = x;
	}

	float getDimension(int i){
		return coordinates[i];
	}
}

public class Obstacle{
	PVector topLeft;
	float gapHeight;
	float width;

	Obstacle(PVector tl, float h, float w){
		topLeft = tl;
		gapHeight = h;
		width = w;
	}

	public float getX(){
		return topLeft.x;
	}

	public float getGapTop(){
		return topLeft.y + gapHeight;
	}

	public float getGapBottom(){
		return topLeft.y;
	}

	public void advance(int x){
		topLeft.x -= x;
	}

	public boolean passed(){
		return ((topLeft.x + width) < 0);
	}
}

// game utilities 

// intersectAABB : check axis-aligned bounding boxes (AABB) for intersection

boolean intersectAABB(float x1, float y1, float w1, float h1, float x2, float y2, float w2, float h2) {
  float left1 = x1;
  float left2 = x2;
  float right1 = x1 + w1;
  float right2 = x2 + w2;

  float top1 = y1;
  float top2 = y2;
  float btm1 = y1 + h1;
  float btm2 = y2 + h2;

  boolean xhit = false;
  boolean yhit = false;

  // overlap
  if (x1 <= x2 && right1 > left2) xhit = true;
  if (x1 > x2 && left1 < right2) xhit = true;

  if (y1 <= y2 && btm1 > top2) yhit = true;
  if (y1 > y2 && top1 < btm2) yhit = true;

  if (xhit && yhit) return true;
  return false;
}