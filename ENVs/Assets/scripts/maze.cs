


using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class MapLocation
{
    public int x;
    public int z;
    
    public MapLocation(int _x, int _z)
    {
        x = _x;
        z = _z;
    }

    public Vector2 ToVector()
    {
        return new Vector2(x, z);
    }

    public static MapLocation operator +(MapLocation a, MapLocation b)
        => new MapLocation(a.x + b.x, a.z + b.z);

    public override bool Equals(object obj)
    {
        if (obj == null || !this.GetType().Equals(obj.GetType()))
        {
            return false;
        }
        else
        {
            return x == ((MapLocation)obj).x && z == ((MapLocation)obj).z;
        }
    }

    public override int GetHashCode()
    {
        // Proper hash code implementation for Dictionary/Set operations
        return x * 31 + z;
    }
}

public class maze : MonoBehaviour
{
    public int width = 10;
    public int height = 10;
    public byte[,] map;
    public int scale = 6;
    public Material wallMaterial;

    public List<MapLocation> directions = new List<MapLocation>
    {
        new MapLocation(1, 0),
        new MapLocation(0, 1),
        new MapLocation(-1, 0),
        new MapLocation(0, -1),
    };

    void Awake()
    {
        // Ensure the maze map is initialized before any other scripts (e.g., Agent) try to use it
        IntializeMap();
        Generate();
        DrawMap();
    }

    void Start()
    {
    }

    void IntializeMap()
    {
        map = new byte[width, height];
        for (int z = 0; z < height; z++)
        {
            for (int x = 0; x < width; x++)
            {
                map[x, z] = 1;
            }
        }
    }

    void Generate()
    {
        // Start from a random position for more variety
        // Ensure starting position is not on the border
        int startX = Random.Range(1, width - 1);
        int startZ = Random.Range(1, height - 1);
        Generate(startX, startZ);
        
        // Ensure all cells are reachable (optional: can be disabled for performance)
        // This guarantees a fully connected maze
        EnsureConnectivity();
    }
     
    void Generate(int x, int z)
    {
        // Boundary check
        if (x <= 0 || x >= width - 1 || z <= 0 || z >= height - 1) return;
        
        // If already visited (empty), skip
        if (map[x, z] == 0) return;
        
        // Prevent loops: don't create a passage if cell already has 2+ empty neighbors
        if (CountSqureNeighbors(x, z) >= 2) return;
        
        // Mark as empty (passage)
        map[x, z] = 0;
        
        // Shuffle directions for randomness
        directions.Shuffle();

        // Recursively generate in all directions
        Generate(x + directions[0].x, z + directions[0].z);
        Generate(x + directions[1].x, z + directions[1].z);
        Generate(x + directions[2].x, z + directions[2].z);
        Generate(x + directions[3].x, z + directions[3].z);
    }
    
    void EnsureConnectivity()
    {
        // First, remove any isolated single-block empty spaces
        // These are empty cells with no empty neighbors - agent would be trapped!
        RemoveIsolatedEmptyCells();
        
        // Optionally extend dead ends to create more paths
        // Only convert walls that have EXACTLY 1 empty neighbor (extends dead ends)
        List<MapLocation> deadEndExtensions = new List<MapLocation>();
        
        for (int z = 1; z < height - 1; z++)
        {
            for (int x = 1; x < width - 1; x++)
            {
                // Only consider walls that have exactly 1 empty neighbor
                // This ensures new passages connect to existing ones
                if (map[x, z] == 1 && CountSqureNeighbors(x, z) == 1)
                {
                    deadEndExtensions.Add(new MapLocation(x, z));
                }
            }
        }
        
        // Convert some dead-end extensions to create alternative paths
        // Convert about 10% to keep maze interesting but more navigable
        int wallsToConvert = Mathf.Max(0, deadEndExtensions.Count / 10);
        for (int i = 0; i < wallsToConvert && i < deadEndExtensions.Count; i++)
        {
            int randomIndex = Random.Range(0, deadEndExtensions.Count);
            MapLocation wall = deadEndExtensions[randomIndex];
            
            // Double-check it still has exactly 1 neighbor (previous conversions might have changed this)
            if (CountSqureNeighbors(wall.x, wall.z) == 1)
            {
                map[wall.x, wall.z] = 0;
            }
            deadEndExtensions.RemoveAt(randomIndex);
        }
        
        // Final cleanup: remove any isolated cells that might have been created
        RemoveIsolatedEmptyCells();
    }
    
    void RemoveIsolatedEmptyCells()
    {
        // Find and fill in any empty cells that have no empty neighbors
        // These would trap the agent with no way out
        bool foundIsolated = true;
        
        // Keep checking until no more isolated cells are found
        // (removing one might reveal another)
        while (foundIsolated)
        {
            foundIsolated = false;
            for (int z = 1; z < height - 1; z++)
            {
                for (int x = 1; x < width - 1; x++)
                {
                    // If it's an empty cell with no empty neighbors, it's isolated
                    if (map[x, z] == 0 && CountSqureNeighbors(x, z) == 0)
                    {
                        map[x, z] = 1; // Convert back to wall
                        foundIsolated = true;
                    }
                }
            }
        }
    }

    void DrawMap()
    {
        for (int z = 0; z < height; z++)
        {
            for (int x = 0; x < width; x++)
            {
                if (map[x, z] == 1)
                {
                    Vector3 pos = new Vector3(x * scale, 0, z * scale);
                    GameObject wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
                    wall.transform.localScale = new Vector3(scale, scale, scale);
                    wall.transform.position = pos;
                    wall.GetComponent<Renderer>().material = wallMaterial;
                    wall.tag = "wall";
                }
            }
        }
    }

    public int CountSqureNeighbors(int x, int z)
    {
        int count = 0;
        if (x <= 0 || x >= width - 1 || z <= 0 || z >= height - 1) return 5;
        if (map[x-1, z] == 0) count++;
        if (map[x+1, z] == 0) count++;
        if (map[x, z-1] == 0) count++;
        if (map[x, z+1] == 0) count++;
        return count;
    }

    // Get a random empty position in the maze (where map value is 0)
    public Vector3 GetRandomEmptyPosition()
    {
        if (map == null)
        {
            Debug.LogError("Maze map is not initialized.");
            return Vector3.zero;
        }
        List<MapLocation> emptyPositions = new List<MapLocation>();
        
        // Collect all empty positions
        for (int z = 0; z < height; z++)
        {
            for (int x = 0; x < width; x++)
            {
                if (map[x, z] == 0)
                {
                    emptyPositions.Add(new MapLocation(x, z));
                }
            }
        }

        if (emptyPositions.Count == 0)
        {
            Debug.LogWarning("No empty positions found in maze!");
            return Vector3.zero;
        }

        // Pick a random empty position
        MapLocation randomPos = emptyPositions[Random.Range(0, emptyPositions.Count)];
        
        // Convert to world position (centered in the cell)
        return new Vector3(randomPos.x * scale, 0.3f, randomPos.z * scale);
    }

    // Check if a world position is inside a wall
    public bool IsPositionInWall(Vector3 worldPosition)
    {
        int x = Mathf.RoundToInt(worldPosition.x / scale);
        int z = Mathf.RoundToInt(worldPosition.z / scale);

        if (x < 0 || x >= width || z < 0 || z >= height)
            return true;

        return map[x, z] == 1;
    }

    // Destroy all existing wall objects
    public void ClearMaze()
    {
        GameObject[] walls = GameObject.FindGameObjectsWithTag("wall");
        foreach (GameObject wall in walls)
        {
            Destroy(wall);
        }
    }

    // Regenerate the entire maze (clear old walls and create new maze)
    public void RegenerateMaze()
    {
        ClearMaze();
        IntializeMap();
        Generate();
        DrawMap();
    }
}