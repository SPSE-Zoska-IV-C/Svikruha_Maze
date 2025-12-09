using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif


public class agent : Agent
{
    [SerializeField] private Transform _goal;
    [SerializeField] private float _moveSpeed = 1.5f;
    [SerializeField] private float _rotationSpeed = 180f;
    [SerializeField] private maze _maze;
    [SerializeField] private Renderer _groundRenderer;
    [SerializeField] private Collider _spawnArea; // Optional area for non-maze environments
    [SerializeField] private float _spawnMargin = 0.75f; // Keep away from edges of spawn area
    [SerializeField] private float _wallCheckRadius = 0.5f; // Clearance from wall colliders
    
    // Ray-based observation settings
    [Header("Ray Observations")]
    [SerializeField] private bool _useRayObservations = true;  // Toggle ray observations
    [SerializeField] private int _numRays = 8;                 // Number of rays (8 directions by default)
    [SerializeField] private float _rayLength = 10f;           // Max ray distance
    [SerializeField] private float _rayStartHeight = 0.5f;     // Height offset for rays
    [SerializeField] private LayerMask _rayLayerMask = -1;     // Layers to detect (default: all)
    
    public Renderer GroundRenderer => _groundRenderer;

    private Renderer _renderer;
    private Rigidbody _rigidbody; // <-- ADDED: For physics-based movement
    
    // State tracking for Python reward calculation
    private float _previousDistanceToGoal;
    private bool _hasHitWall = false;
    private bool _hasReachedGoal = false;
    private float _timeInWall = 0f;

    [HideInInspector]public int _currentEpisode = 0;
    [HideInInspector]public float _cumulativeReward = 0f;
    [HideInInspector]public int _currentStep = 0;
    [HideInInspector]public int _collisionCount = 0;
    
    public override void Initialize()
    {
        Debug.Log("Initialize"); 

        _rigidbody = GetComponent<Rigidbody>(); // <-- ADDED: Get the Rigidbody component
        _renderer =  GetComponent<Renderer>();
        _currentEpisode = 0;
        _cumulativeReward = 0f;
        _currentStep = 0;
    }

    public override void OnEpisodeBegin()
    {
        Debug.Log("OnEpisodeBegin");
        _currentEpisode++;
        _cumulativeReward = 0f;
        _currentStep = 0;
        _collisionCount = 0;
        _hasHitWall = false;
        _hasReachedGoal = false;
        _timeInWall = 0f;
        _renderer.material.color = Color.blue;

        // Regenerate maze every 100 episodes for better generalization
        if (_maze != null && _currentEpisode % 100 == 0)
        {
            Debug.Log($"Regenerating maze at episode {_currentEpisode}");
            _maze.RegenerateMaze();
        }

        SpawnObject();
        
        // Set the initial distance for Python reward calculation
        _previousDistanceToGoal = Vector3.Distance(transform.position, _goal.position); 
    }

    private void SpawnObject()
    {
        // If a maze is assigned, use maze-based spawning
        if (_maze != null)
        {
            Vector3 agentPosition = _maze.GetRandomEmptyPosition();
            transform.position = agentPosition;
            transform.localRotation = Quaternion.identity;

            Vector3 goalPosition;
            int maxAttempts = 100;
            int attempts = 0;

            do
            {
                goalPosition = _maze.GetRandomEmptyPosition();
                attempts++;
            } while (Vector3.Distance(agentPosition, goalPosition) < 2f && attempts < maxAttempts);

            _goal.position = goalPosition;
            return;
        }

        // Otherwise, fall back to spawning inside a provided collider area (e.g., a Plane's collider)
        if (_spawnArea == null)
        {
            Debug.LogError("Assign either a Maze or a Spawn Area collider on the agent!");
            return;
        }

        Bounds bounds = _spawnArea.bounds;

        bool TryGetClearPoint(out Vector3 point, int maxAttempts = 100)
        {
            for (int i = 0; i < maxAttempts; i++)
            {
                float x = Random.Range(bounds.min.x + _spawnMargin, bounds.max.x - _spawnMargin);
                float z = Random.Range(bounds.min.z + _spawnMargin, bounds.max.z - _spawnMargin);
                Vector3 candidate = new Vector3(x, bounds.max.y + 2f, z);

                // The first collider below must be the spawn area (ground), not a wall or other object
                if (Physics.Raycast(candidate, Vector3.down, out RaycastHit hitInfo, 10f))
                {
                    if (hitInfo.collider != _spawnArea)
                    {
                        continue; // something else (likely a wall) is above the ground here
                    }

                    // place slightly above the ground contact point
                    candidate.y = hitInfo.point.y + 0.3f;
                }
                else
                {
                    continue; // nothing below, skip
                }

                // Check clearance from walls
                Collider[] hits = Physics.OverlapSphere(candidate, _wallCheckRadius);
                bool touchesWall = false;
                for (int h = 0; h < hits.Length; h++)
                {
                    if (hits[h] != null && hits[h].CompareTag("wall"))
                    {
                        touchesWall = true;
                        break;
                    }
                }
                if (!touchesWall)
                {
                    point = candidate;
                    return true;
                }
            }
            point = Vector3.zero;
            return false;
        }

        // Agent position
        if (!TryGetClearPoint(out Vector3 nonMazeAgentPos))
        {
            Debug.LogWarning("Failed to find a clear spawn for Agent in spawn area; using center.");
            nonMazeAgentPos = bounds.center + new Vector3(0f, 0.3f - bounds.extents.y, 0f);
        }
        transform.position = nonMazeAgentPos;
        transform.localRotation = Quaternion.identity;

        // Goal position
        Vector3 nonMazeGoalPos = nonMazeAgentPos;
        int nonMazeAttempts = 0;
        const int nonMazeMaxAttempts = 200;
        while (nonMazeAttempts < nonMazeMaxAttempts)
        {
            if (TryGetClearPoint(out nonMazeGoalPos) && Vector3.Distance(nonMazeAgentPos, nonMazeGoalPos) >= 2f)
            {
                break;
            }
            nonMazeAttempts++;
        }
        _goal.position = nonMazeGoalPos;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // --- OBSERVATIONS FOR AGENT DECISION MAKING ---
        
        // Calculate the goal's position relative to the agent's position AND rotation
        // This tells the agent "how far forward/back" (Z) and "how far left/right" (X) the goal is.
        Vector3 relativeGoalPosition = transform.InverseTransformPoint(_goal.position);

        // Normalize these values (dividing by 5f as you had before, adjust if your area is larger/smaller)
        
        // Observation 1: Goal's local X position (left/right)
        sensor.AddObservation(relativeGoalPosition.x / 5f);
        
        // Observation 2: Goal's local Z position (forward/back)
        sensor.AddObservation(relativeGoalPosition.z / 5f);

        // --- ADDITIONAL OBSERVATIONS FOR PYTHON REWARD CALCULATION ---
        
        // Observation 3: Current distance to goal (normalized)
        float currentDistance = Vector3.Distance(transform.position, _goal.position);
        sensor.AddObservation(currentDistance / 10f); // Normalize by max expected distance
        
        // Observation 4: Previous distance to goal (normalized) - for reward shaping
        sensor.AddObservation(_previousDistanceToGoal / 10f);
        
        // Observation 5: Has hit wall (boolean as float: 1.0 = true, 0.0 = false)
        sensor.AddObservation(_hasHitWall ? 1.0f : 0.0f);
        
        // Observation 6: Has reached goal (boolean as float)
        sensor.AddObservation(_hasReachedGoal ? 1.0f : 0.0f);
        
        // Observation 7: Time spent in wall collision (normalized)
        sensor.AddObservation(_timeInWall);
        
        // Observation 8: Agent velocity magnitude (for potential reward shaping)
        if (_rigidbody != null)
        {
            sensor.AddObservation(_rigidbody.linearVelocity.magnitude / 5f); // Normalize by max expected speed
        }
        else
        {
            sensor.AddObservation(0f);
        }

        // --- RAY-BASED OBSERVATIONS FOR WALL DETECTION ---
        // These help the agent perceive walls/obstacles in multiple directions
        // Observations 9 to 9+numRays: Distance to nearest obstacle in each direction (normalized)
        // Observations 9+numRays to 9+2*numRays: Whether each ray hit a wall (0 or 1)
        if (_useRayObservations)
        {
            CollectRayObservations(sensor);
        }

        // --- UPDATE CUMULATIVE REWARD FOR VISUALIZATION ONLY ---
        // Since rewards are calculated in Python, we replicate the logic here to show it in the GUI.
        // This does NOT affect training if using Python-side reward calculation.
        
        if (_currentStep > 0)
        {
            float stepReward = 0f;

            // 1. Goal Reached
            if (_hasReachedGoal)
            {
                stepReward += 1.0f;
            }

            // 2. Distance Improvement
            float distanceDelta = _previousDistanceToGoal - currentDistance;
            stepReward += distanceDelta * 0.01f;

            // 3. Wall Collision
            if (_hasHitWall)
            {
                stepReward += -0.15f;
                stepReward += _timeInWall * -0.01f;
            }

            // 4. Time Penalty
            float maxSteps = MaxStep > 0 ? MaxStep : 1000f;
            stepReward += -2.0f / maxSteps;

            _cumulativeReward += stepReward;
        }

        _previousDistanceToGoal = currentDistance;
    }

    /// <summary>
    /// Collects ray-based observations for wall/obstacle detection.
    /// Casts rays in multiple directions and reports distances and hit types.
    /// </summary>
    private void CollectRayObservations(VectorSensor sensor)
    {
        Vector3 rayOrigin = transform.position + Vector3.up * _rayStartHeight;
        float angleStep = 360f / _numRays;
        
        for (int i = 0; i < _numRays; i++)
        {
            // Calculate ray direction (evenly distributed around the agent)
            // Start from forward direction and go clockwise
            float angle = i * angleStep;
            Vector3 direction = Quaternion.Euler(0f, angle, 0f) * transform.forward;
            
            // Perform raycast
            bool hitSomething = Physics.Raycast(rayOrigin, direction, out RaycastHit hit, _rayLength, _rayLayerMask);
            
            if (hitSomething)
            {
                // Normalized distance (0 = hit at origin, 1 = hit at max range or no hit)
                float normalizedDistance = hit.distance / _rayLength;
                sensor.AddObservation(normalizedDistance);
                
                // Check if we hit a wall specifically
                bool isWall = hit.collider.CompareTag("wall");
                sensor.AddObservation(isWall ? 1.0f : 0.5f);  // 1.0 = wall, 0.5 = other obstacle
            }
            else
            {
                // No hit - report max distance and no wall
                sensor.AddObservation(1.0f);  // Max normalized distance
                sensor.AddObservation(0.0f);  // No obstacle detected
            }
            
            // Debug visualization (only in editor)
            #if UNITY_EDITOR
            Color rayColor = hitSomething ? (hit.collider.CompareTag("wall") ? Color.red : Color.yellow) : Color.green;
            float drawDistance = hitSomething ? hit.distance : _rayLength;
            Debug.DrawRay(rayOrigin, direction * drawDistance, rayColor);
            #endif
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        discreteActionsOut[0] = 0;
#if ENABLE_INPUT_SYSTEM
        var keyboard = Keyboard.current;
        if (keyboard != null)
        {
            if (keyboard.wKey.isPressed) discreteActionsOut[0] = 1;
            if (keyboard.dKey.isPressed) discreteActionsOut[0] = 3;
            if (keyboard.aKey.isPressed) discreteActionsOut[0] = 2;
        }
#else
        if (Input.GetKey(KeyCode.W)) { discreteActionsOut[0] = 1; }
        if (Input.GetKey(KeyCode.D)) { discreteActionsOut[0] = 3; }
        if (Input.GetKey(KeyCode.A)) { discreteActionsOut[0] = 2; }
#endif
    }
        
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // Capture previous distance BEFORE moving to ensure precise reward calculation tracking
        // _previousDistanceToGoal = Vector3.Distance(transform.position, _goal.position);
        // MOVED TO CollectObservations for cleaner reward tracking

        MoveAgent(actionBuffers.DiscreteActions);

        // Update step counter
        _currentStep++;
        
        // Update cumulative reward from Python (if set via SetReward)
        // _cumulativeReward = GetCumulativeReward();
        // DISABLED: We are calculating _cumulativeReward manually in CollectObservations for visualization
        
        // NOTE: All rewards are now calculated in Python based on observations
        // Python will call SetReward() via the low-level API if needed
    }

    public void MoveAgent(ActionSegment<int> act)
    {
        // --- MODIFIED: Use Rigidbody for movement ---
        var action = act[0];
        
        // Use Time.fixedDeltaTime because physics operations happen in FixedUpdate
        float deltaTime = Time.fixedDeltaTime; 

        switch (action)
        {
            case 1:
                // Move forward using Rigidbody.MovePosition
                Vector3 newPos = transform.position + transform.forward * _moveSpeed * deltaTime;
                _rigidbody.MovePosition(newPos);
                break;
            case 2:
                // Rotate left using Rigidbody.MoveRotation
                Quaternion newRotLeft = Quaternion.Euler(0f, -_rotationSpeed * deltaTime, 0f);
                _rigidbody.MoveRotation(_rigidbody.rotation * newRotLeft);
                break;
            case 3:
                // Rotate right using Rigidbody.MoveRotation
                Quaternion newRotRight = Quaternion.Euler(0f, _rotationSpeed * deltaTime, 0f);
                _rigidbody.MoveRotation(_rigidbody.rotation * newRotRight);
                break;
        }
        // Case 0 (do nothing) is handled automatically
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.CompareTag("goal"))
        {
            GoalReached();
        }
    }

    private void GoalReached()
    {
        _hasReachedGoal = true;
        // Reward will be calculated in Python based on observation
        // Python will detect _hasReachedGoal = 1.0 in observations and give reward
        
        EndEpisode(); // End the episode and start a new one
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("wall"))
        {
            _hasHitWall = true;
            _timeInWall = 0f;
            _collisionCount++;
            
            // Visual feedback
            if (_renderer != null)
            {
                _renderer.material.color = Color.red;
            }
            
            // Reward penalty will be calculated in Python based on observation
        }
    }

    private void OnCollisionStay(Collision collision)
    {
        if (collision.gameObject.CompareTag("wall"))
        {
            _hasHitWall = true;
            _timeInWall += Time.fixedDeltaTime;
            
            // Reward penalty will be calculated in Python based on _timeInWall observation
        }
    }

    private void OnCollisionExit(Collision collision)
    {
        if (collision.gameObject.CompareTag("wall"))
        {
            _hasHitWall = false;
            _timeInWall = 0f;
            
            // Visual feedback
            if (_renderer != null)
            {
                _renderer.material.color = Color.blue;
            }
        }
    }
}