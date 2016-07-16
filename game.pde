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
	int obstacleSpeed = 3;
  	
  	int difficulty = 50; // larger is easier
  	int playerWidth = 12;
  	int playerHeight = 10;

  	int highScore = 0;
  	int episodes = 0;
  	int points = 0;

  	boolean terminal = false;

  	PVector position, velocity, acceleration;

	FlappyBird(PhysicsModel w){
		world = w;
		
		reset();
	}

	public void nextFrame(){
		if(terminal){
			reset();
			terminal = false;
		}

		// player
		applyForces();
		velocity.y += acceleration.y;
		position.y += velocity.y;
		
		// obstacles
		for (int i = 0; i < obstacles.size(); i++) {
		  Obstacle o = obstacles.get(i);
		  o.advance(obstacleSpeed);
		  if(o.passed()){
		  	obstacles.remove(i);
		  	points++;
		  }
		}


		Obstacle last = obstacles.get(obstacles.size()-1);
		if(last.topLeft.x + obstacleWidth + obstacleGap <= stageWidth) addObstacleAt(stageWidth);
		
		// collisions

		Obstacle first = obstacles.get(0);

		if(position.y + playerWidth/2 > stageHeight) gameOver();
		if(position.y - playerWidth/2 < 0) gameOver();
		//upper
		if(intersectAABB(position.x,position.y-playerHeight/2, playerWidth, playerHeight,first.topLeft.x,0,obstacleWidth,first.topLeft.y-first.gapHeight)) gameOver();
		//lower
		if(intersectAABB(position.x,position.y-playerHeight/2, playerWidth, playerHeight,first.topLeft.x,first.topLeft.y,obstacleWidth,stageHeight-first.topLeft.y)) gameOver();
			
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
		State s = new State();
		Obstacle o = obstacles.get(0);
		s.setDimension(0, o.topLeft.x - playerWidth); // player distance to obstacle
		s.setDimension(1, o.gapHeight); // player distance to lower pipe
		s.setDimension(2, o.topLeft.y - (position.y + playerHeight/2)); // player distance to lower pipe
		// s.setDimension(2, (position.y - playerHeight/2) - (o.topLeft.y - o.gapHeight)); // player distance to upper pipe
		s.setDimension(3, velocity.y); // player velocity
		return s;
	}

	public void draw(){
		noStroke();
		fill(#CC0000);
		rect(position.x, position.y-playerHeight/2, playerWidth, playerHeight);
		fill(#990000);
		rect(position.x, position.y-playerHeight/4, playerWidth*3/4, playerHeight*3/4 - 1);
		fill(#EAB804);
		rect(position.x + playerWidth - 2, position.y - 2, 3, 2);


		for (int i = 0; i < obstacles.size(); i++) {
		  Obstacle o = obstacles.get(i);
		  
		  //upper
		  fill(#11AA11);
		  rect(o.topLeft.x, 0, obstacleWidth, o.topLeft.y - o.gapHeight);
		  
		  fill(#1FAF1F);
		  rect(o.topLeft.x + 1, 0, obstacleWidth - 2, o.topLeft.y - o.gapHeight);
		  
		  fill(#22BB22);
		  rect(o.topLeft.x + 3, 0, obstacleWidth - 6, o.topLeft.y - o.gapHeight);

		  fill(#00AA00);
		  rect(o.topLeft.x, o.topLeft.y - o.gapHeight - 10, obstacleWidth, 10);
		  
		  //lower

		  fill(#11AA11);
		  rect(o.topLeft.x, o.topLeft.y + 10, obstacleWidth, stageHeight - o.topLeft.y - 10);
		  
		  fill(#1FAF1F);
		  rect(o.topLeft.x + 1, o.topLeft.y + 10, obstacleWidth - 2, stageHeight - o.topLeft.y - 10);
		  
		  fill(#22BB22);
		  rect(o.topLeft.x + 3, o.topLeft.y + 10, obstacleWidth - 6, stageHeight - o.topLeft.y - 10);

		  fill(#00AA00);
		  rect(o.topLeft.x, o.topLeft.y, obstacleWidth, 10);
		}

		textAlign(CENTER,CENTER);
		textSize(51);
		fill(#AAAAAA);
		text(points, stageWidth/2, stageHeight/2 + 1);
		textSize(50);
		fill(#EEEEEE);
		text(points, stageWidth/2, stageHeight/2);
	}

	// internal game methods

	private void gameOver(){
		terminal = true;
	}

	private void reset(){
		obstacles.clear(); // clear old game objects		

		// reset game statistics
		episodes++;
		highScore = max(points,highScore);
		points = 0;

		// reset game physics
		position = new PVector(0, stageHeight/2);
		velocity = new PVector(0,0);
		acceleration = new PVector(0,0);

		// add new objects
		for(int i = 0; i < floor(stageWidth / (obstacleGap + obstacleWidth)); i++){
			addObstacleAt(obstacleGap + (obstacleGap + obstacleWidth) * i);	
		}
	}

	private void addObstacleAt(int x){
		PVector gapCenter = new PVector(x, position.y + random(-difficulty, difficulty));
		int gapHeight = difficulty + round(random(difficulty));

		// topLeft is top left coordinate of lower pipe
		PVector topLeft = gapCenter.copy();
		topLeft.y += gapHeight/2;

		// limit pipes to at least 20 on top / bottom
		topLeft.y = min(topLeft.y, stageHeight - 20);
		if(topLeft.y - gapHeight < 20) topLeft.y += 20 - (topLeft.y - gapHeight);

		Obstacle o = new Obstacle(topLeft, gapHeight, obstacleWidth);
		obstacles.add(o);
	}

	private void applyForces(){
		switch(world){
			case MOVE:
				break;
			case JUMP:
				acceleration.y = 0.25;
				break;
			case HOVER:
				acceleration.y += 0.1;
				break;
		}
	}

	private void jump(){
		switch(world){
			case MOVE:
				position.y -= 10;
				break;
			case JUMP:
				// if(velocity.y > 0) velocity.y -= 7;
				velocity.y = -5; // flappy bird is not the real life
				break;
			case HOVER:
				break;
		}
	}

	private void fall(){
		switch(world){
			case MOVE:
				position.y += 10;
				break;
			case JUMP:
				break;
			case HOVER:
				break;
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

	State(State s){
		this.coordinates = s.coordinates;
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