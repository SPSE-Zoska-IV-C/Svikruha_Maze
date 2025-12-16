using UnityEngine;

/// <summary>
/// Training optimization script that reduces GPU load during ML training.
/// Attach this to any GameObject in the scene (e.g., Ground, Main Camera, or empty Training object).
/// 
/// Problem: When time_scale=20.0 in Python, Unity tries to render 60*20 = 1200 FPS,
/// causing GPU overload (87%+ usage) and lag.
/// 
/// Solution: Limit rendering to 30 FPS while physics still runs at high speed.
/// This saves ~90% of GPU power for other tasks.
/// </summary>
public class TrainingManager : MonoBehaviour
{
    [Header("Training Settings")]
    [Tooltip("Target frame rate for rendering. Lower = less GPU usage. 30 is good for training.")]
    [SerializeField] private int _targetFrameRate = 60;
    
    [Tooltip("Disable VSync to ensure targetFrameRate works immediately")]
    [SerializeField] private bool _disableVSync = true;
    
    [Tooltip("Enable this for maximum training speed (disables some visual features)")]
    [SerializeField] private bool _optimizeForTraining = true;
    
    [Header("Physics Optimization")]
    [Tooltip("Fixed timestep for physics (higher = fewer physics calculations). Default is 0.02 (50Hz). Try 0.04 for training.")]
    [SerializeField] private float _fixedTimestep = 0.02f;  // Keep default, let time_scale handle speed
    
    [Tooltip("Maximum time physics can simulate per frame (prevents spiral of death)")]
    [SerializeField] private float _maximumDeltaTime = 0.1f;

    [Header("Debug")]
    [SerializeField] private bool _showFPS = false;
    
    private float _deltaTime = 0f;
    
    void Awake()
    {
        // Limit rendering frame rate
        Application.targetFrameRate = _targetFrameRate;
        
        // Disable VSync to ensure targetFrameRate takes effect immediately
        if (_disableVSync)
        {
            QualitySettings.vSyncCount = 0;
        }
        
        // CRITICAL: Limit how much physics can run per frame
        // This prevents the "spiral of death" where physics falls behind
        Time.maximumDeltaTime = _maximumDeltaTime;
        
        // Optionally adjust fixed timestep (be careful - affects agent behavior)
        if (_fixedTimestep > 0)
        {
            Time.fixedDeltaTime = _fixedTimestep;
        }
        
        // Additional optimizations for training
        if (_optimizeForTraining)
        {
            // Reduce shadow quality (saves GPU)
            QualitySettings.shadows = ShadowQuality.Disable;
            
            // Disable soft particles
            QualitySettings.softParticles = false;
            
            // Reduce texture quality (0 = full, 1 = half, 2 = quarter, etc.)
            QualitySettings.globalTextureMipmapLimit = 2;
            
            // Disable anti-aliasing
            QualitySettings.antiAliasing = 0;
            
            // Reduce skin weights
            QualitySettings.skinWeights = SkinWeights.OneBone;
            
            // Disable real-time reflection probes
            QualitySettings.realtimeReflectionProbes = false;
        }
        
        Debug.Log($"[TrainingManager] Initialized: targetFPS={_targetFrameRate}, fixedDeltaTime={Time.fixedDeltaTime}, maxDeltaTime={Time.maximumDeltaTime}");
    }
    
    void Update()
    {
        // FPS counter for debugging
        if (_showFPS)
        {
            _deltaTime += (Time.unscaledDeltaTime - _deltaTime) * 0.1f;
        }
    }
    
    void OnGUI()
    {
        if (_showFPS)
        {
            float fps = 1.0f / _deltaTime;
            string text = $"FPS: {fps:0.} | TimeScale: {Time.timeScale:0.0}x";
            
            GUIStyle style = new GUIStyle();
            style.fontSize = 20;
            style.normal.textColor = fps < 20 ? Color.red : (fps < 40 ? Color.yellow : Color.green);
            
            // Positioned below GUI_Agent elements (which use y=20, 60, 100)
            GUI.Label(new Rect(20, 140, 300, 30), text, style);
        }
    }
    
    /// <summary>
    /// Call this method to switch between training and visualization modes.
    /// </summary>
    public void SetTrainingMode(bool isTraining)
    {
        if (isTraining)
        {
            Application.targetFrameRate = _targetFrameRate;
            QualitySettings.shadows = ShadowQuality.Disable;
        }
        else
        {
            Application.targetFrameRate = 60; // Normal playback
            QualitySettings.shadows = ShadowQuality.All;
        }
    }
}
