using UnityEngine;

[ExecuteAlways]
public class FitCameraToPlane : MonoBehaviour
{
    [SerializeField] private Renderer target;   // Plane's MeshRenderer
    [SerializeField] private float padding = 0.5f;
    [SerializeField] private bool topDownOrtho = true;

    private Camera cam;

    void OnEnable()
    {
        cam = GetComponent<Camera>();
    }

    void LateUpdate()
    {
        if (!cam || !target) return;

        Bounds b = target.bounds;
        Vector3 c = b.center;

        if (topDownOrtho)
        {
            cam.orthographic = true;
            transform.position = new Vector3(c.x, b.max.y + 5f, c.z);
            transform.rotation = Quaternion.Euler(90f, 0f, 0f);

            float halfW = b.extents.x + padding;
            float halfH = b.extents.z + padding;
            cam.orthographicSize = Mathf.Max(halfH, halfW / cam.aspect);
        }
        else
        {
            cam.orthographic = false; // angled perspective option
            float pitch = 55f, yaw = 45f;
            transform.rotation = Quaternion.Euler(pitch, yaw, 0f);

            float halfW = b.extents.x + padding;
            float halfH = b.extents.z + padding;
            float vFov = cam.fieldOfView * Mathf.Deg2Rad;
            float hFov = Camera.VerticalToHorizontalFieldOfView(cam.fieldOfView, cam.aspect) * Mathf.Deg2Rad;
            float dist = Mathf.Max(halfH / Mathf.Tan(vFov / 2f), halfW / Mathf.Tan(hFov / 2f));

            transform.position = c - transform.forward * dist + Vector3.up * (b.extents.y + padding);
        }
    }
}