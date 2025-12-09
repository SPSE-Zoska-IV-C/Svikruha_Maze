using UnityEngine;

public class GUI_Agent : MonoBehaviour
{
    [SerializeField] private agent _agent;
    private GUIStyle _defaultStyle = new GUIStyle();
    private GUIStyle _positiveStyle = new GUIStyle();
    private GUIStyle _negativeStyle = new GUIStyle();

    void Start()
    {
        if (_agent == null)
        {
            _agent = GetComponent<agent>();
            if (_agent == null)
            {
                _agent = FindFirstObjectByType<agent>();
            }
        }
        
        _defaultStyle.fontSize = 20;
        _defaultStyle.normal.textColor = Color.yellow;

        _positiveStyle.fontSize = 20;
        _positiveStyle.normal.textColor = Color.green;

        _negativeStyle.fontSize = 20;
        _negativeStyle.normal.textColor = Color.red;
    }

    private void OnGUI()
    {
        if (_agent == null)
        {
            GUI.Label(new Rect(20, 20, 600, 30), "GUI_Agent: assign an Agent reference.", _defaultStyle);
            return;
        }

        string debugEpisode = "Episode: " + _agent._currentEpisode + " - Step: " + _agent._currentStep;
        string debugReward = "Reward: " + _agent._cumulativeReward.ToString("F2");
        string debugCollisions = "Collisions: " + _agent._collisionCount;

        GUIStyle rewardStyle = _agent._cumulativeReward < 0 ? _negativeStyle : _positiveStyle;
        GUIStyle collisionStyle = _agent._collisionCount > 0 ? _negativeStyle : _positiveStyle;

        GUI.Label(new Rect(20, 20, 500, 30), debugEpisode, _defaultStyle);
        GUI.Label(new Rect(20, 60, 500, 30), debugReward, rewardStyle);
        GUI.Label(new Rect(20, 100, 500, 30), debugCollisions, collisionStyle);
    }
}
