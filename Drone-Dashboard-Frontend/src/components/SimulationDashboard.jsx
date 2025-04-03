import React, { useEffect, useState } from "react";
import { Maximize2, Monitor, Camera } from "lucide-react";

const SERVER_WS_URL = "ws://localhost:8765";
const SIMULATION_URL = "http://127.0.0.1:5500/Autonomous-Search-and-Rescue-Drone/Drone-UI/index.html";

const DEFAULT_IMAGE = "/test_data/thermal/no_human/FLIR_04123_jpeg_jpg.rf.fa8691c9bfaccaa604ff9cbb7f1af48c.jpg";

const SimulationDashboard = () => {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [ws, setWs] = useState(null);
  const [detectedImage, setDetectedImage] = useState(null);
  const [isDetecting, setIsDetecting] = useState(false);

  useEffect(() => {
    const socket = new WebSocket(SERVER_WS_URL);

    socket.onopen = () => {
      console.log("✅ Connected to WebSocket Server");
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("📩 Received:", data);

      if (data.type === "thermal") {
        setDetectedImage(data.image);
        setIsDetecting(false);
        
        // Clear detection after 4 seconds
        setTimeout(() => {
          setDetectedImage(null);
        }, 4000);
      } else if (data.type === "audio") {
        alert(`Audio Detection Result: ${data.result}`);
      }
    };

    socket.onerror = (error) => console.error("❌ WebSocket Error:", error);
    socket.onclose = () => console.log("🔌 WebSocket Disconnected");

    setWs(socket);
    return () => socket.close();
  }, []);

  const sendDetectionRequest = () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ model: "thermal" }));
      console.log("📤 Sent detection request for thermal");
      setIsDetecting(true);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 p-6 text-gray-100">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className={`bg-gray-800 rounded-lg border border-gray-700 shadow-xl 
          ${isFullscreen ? "fixed inset-0 z-50" : ""} 
          transition-all duration-300 hover:border-blue-500`}
        >
          <div className="flex items-center justify-between p-4 border-b border-gray-700 bg-gray-800/50 backdrop-blur">
            <div className="flex items-center gap-2">
              <Monitor className="h-5 w-5 text-blue-400" />
              <h2 className="text-xl font-semibold">3D Simulation</h2>
            </div>
            <button
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="p-2 hover:bg-gray-700 rounded-full transition-all duration-300 hover:text-blue-400 hover:rotate-90"
            >
              <Maximize2 className="h-5 w-5" />
            </button>
          </div>
          <div className="p-4">
            {isLoading && (
              <div className="absolute inset-0 flex items-center justify-center bg-gray-800/80">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400"></div>
              </div>
            )}
            <iframe
              src={SIMULATION_URL}
              className={`w-full bg-black rounded-lg ${isFullscreen ? "h-[calc(100vh-120px)]" : "h-[60vh]"} transition-all duration-300`}
              title="Simulation Screen"
              onLoad={() => setIsLoading(false)}
            />
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg border border-gray-700 shadow-xl transition-all duration-300 hover:border-blue-500">
          <div className="p-4 border-b border-gray-700">
            <div className="flex items-center gap-2">
              <Camera className="h-5 w-5 text-blue-400" />
              <h2 className="text-xl font-semibold">Box</h2>
            </div>
          </div>
          <div className="p-4">
            <div className="relative">
              <div className="absolute top-2 right-2 z-10">
                <button
                  className={`px-3 py-1 rounded text-sm ${
                    isDetecting ? "bg-blue-500 text-white" : "bg-gray-700 text-white hover:bg-blue-500"
                  }`}
                  onClick={sendDetectionRequest}
                  disabled={isDetecting}
                >
                  {isDetecting ? "Detecting..." : "Detect"}
                </button>
              </div>
              <div className="aspect-video rounded-lg border border-gray-700 overflow-hidden transition-all duration-300 hover:border-blue-500">
                {detectedImage ? (
                  <img
                    src={`data:image/png;base64,${detectedImage}`}
                    alt="Thermal detection"
                    className="w-full h-full object-cover rounded-lg transition-all duration-500 hover:scale-110"
                  />
                ) : (
                  <img
                    src={DEFAULT_IMAGE}
                    alt="Default"
                    className="w-full h-full object-cover rounded-lg opacity-50"
                  />
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SimulationDashboard;
