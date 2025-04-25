using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using TMPro;  // For using TextMeshPro
using UnityEngine;

public class UDPReceiver : MonoBehaviour
{
    UdpClient udpClient;
    Thread receiveThread;
    public TextMeshProUGUI alertText;  // Assign this in the inspector

    void Start()
    {
        udpClient = new UdpClient(8052);  // Port must match Python sender
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    void ReceiveData()
    {
        while (true)
        {
            try
            {
                IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Any, 0);
                byte[] data = udpClient.Receive(ref remoteEndPoint);
                string message = Encoding.UTF8.GetString(data);
                Debug.Log("Received UDP message: " + message);

                if (message.StartsWith("ALERT|"))
                {
                    string alert = message.Split('|')[1];

                    // Only one call to UnityMainThreadDispatcher
                    UnityMainThreadDispatcher.Instance().Enqueue(() =>
                    {
                        UpdateAlertUI(alert);
                    });
                }
            }
            catch (System.Exception e)
            {
                Debug.Log("UDP Receive Error: " + e.Message);
            }
        }
    }

    void UpdateAlertUI(string alert)
    {
        alertText.text = $"ALERT: COGNITIVE LOAD {alert}";
        alertText.color = Color.red;
    }

    void OnApplicationQuit()
    {
        udpClient.Close();
        receiveThread.Abort();
    }
}
