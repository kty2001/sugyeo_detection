// SugyeoPage.js
import React, { useState, useEffect } from "react";

export default function StreamingPage() {
  const [alert, setAlert] = useState(false);
  const [cameraConnected, setCameraConnected] = useState(null); // null = 초기 상태
  const [reloadKey, setReloadKey] = useState(0);
  const [cameras, setCameras] = useState([]); // 연결 가능한 카메라 목록
  const [selectedCamera, setSelectedCamera] = useState(""); // 선택된 카메라
  const [logs, setLogs] = useState([]); // 기록 저장
  const [showLogs, setShowLogs] = useState(false); // 기록 표시 여부
  const [streaming, setStreaming] = useState(false); // 시작/중지 상태

  // 카메라 목록 불러오기
  useEffect(() => {
    const loadCameras = async () => {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter((d) => d.kind === "videoinput");
        setCameras(videoDevices);
        if (videoDevices.length > 0) {
          setSelectedCamera(videoDevices[0].deviceId);
        }
      } catch (err) {
        console.error("카메라 목록 불러오기 실패:", err);
      }
    };

    loadCameras();
  }, []);

  // 주기적으로 백엔드 상태 확인
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch("http://localhost:8000/api/sugyeo/check");
        const data = await res.json();
        setAlert(data.alert);
      } catch (err) {
        console.error("경고 상태 확인 실패:", err);
      }
    }, 1000);
    return () => clearInterval(interval);
  }, []);

const handleToggleStream = async () => {
  if (!streaming) {
    const cameraIndex = cameras.findIndex(cam => cam.deviceId === selectedCamera)
    await fetch(`http://localhost:8000/api/sugyeo/start?camera=${cameraIndex}`, {
      method: "POST"
    })
    setReloadKey(prev => prev + 1)
  } else {
    const cameraIndex = cameras.findIndex(cam => cam.deviceId === selectedCamera)
    await fetch(`http://localhost:8000/api/sugyeo/stop?camera=${cameraIndex}`, {
      method: "POST"
    })
  }
  setStreaming(prev => !prev)
}

  // 연결 재시도
  const handleRetry = () => {
    setCameraConnected(null); // 로딩 상태로
    setReloadKey((prev) => prev + 1);
  };

  // 스트림 접근 시도 (선택된 카메라 기반)
  useEffect(() => {
    if (!streaming) return;

    const testConnection = async () => {
      try {
        // 서버에는 index 전달
        const cameraIndex = cameras.findIndex(
          (cam) => cam.deviceId === selectedCamera
        );
        const res = await fetch(
          `http://localhost:8000/api/sugyeo/process?reload=${reloadKey}&camera=${cameraIndex}`,
          { method: "GET" }
        );
        if (res.ok) {
          setCameraConnected(true);
        } else {
          setCameraConnected(false);
        }
      } catch {
        setCameraConnected(false);
      }
    };

    if (selectedCamera) testConnection();
  }, [reloadKey, selectedCamera, cameras, streaming]);

  // 기록 보기 버튼 클릭
  const handleViewLogs = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/sugyeo/logs");
      const data = await res.json();
      setLogs(Array.isArray(data.logs) ? data.logs : []);
      setShowLogs(true);
    } catch (err) {
      console.error("기록 불러오기 실패:", err);
      setLogs([]);
    }
  };

  const handleSaveCSV = () => {
    if (!logs.length) return;

    const header = ["시간", "경고", "전체 사람", "오버랩", "오버랩 아님"];
    const rows = logs.map((log) => [
      log.timestamp,
      log.alert ? "⚠️" : "",
      log.total_people,
      log.overlap_count,
      log.non_overlap_count,
    ]);

    const csvContent =
      [header, ...rows].map((e) => e.join(",")).join("\n");
    const blob = new Blob(["\uFEFF" + csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "frame_logs.csv";
    link.click();
    URL.revokeObjectURL(url);
  };

  
  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-6">
      <h1 className="text-3xl font-bold text-gray-800 mb-6">
        YOLO + Segformer Live Stream
      </h1>

      <div
        className={`w-full max-w-4xl mb-xl rounded-2xl shadow-lg p-4 transition-colors duration-300 ${
          alert ? "border-4 border-red-500" : "border-4 border-green-500"
        }`}
      >
        <div className="flex justify-center items-center bg-black rounded-xl border border-gray-300 h-[400px] w-full overflow-hidden">
          {streaming ? (
            cameraConnected === null ? (
              <p className="text-gray-400 text-lg animate-pulse">🎥 카메라 연결 확인 중...</p>
            ) : cameraConnected ? (
              <img
                key={reloadKey}
                src={`http://localhost:8000/api/sugyeo/process?reload=${reloadKey}&camera=${cameras.findIndex(cam => cam.deviceId === selectedCamera)}`}
                alt="Live Stream"
                className="rounded-xl h-full w-full object-contain"
                onError={() => setCameraConnected(false)}
                onLoad={() => setCameraConnected(true)}
              />
            ) : (
              <div className="flex flex-col items-center justify-center text-white text-lg">
                <p>📷 카메라 연결 안 됨</p>
                <p className="text-sm text-gray-400 mt-2">
                  카메라를 연결하거나 서버를 확인하세요.
                </p>
                <button
                  onClick={handleRetry}
                  className="mt-4 bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg transition"
                >
                  연결 재시도
                </button>
              </div>
            )
          ) : (
            <div className="flex justify-center items-center text-gray-400 h-full w-full">
              🎥 스트리밍이 중지됨
            </div>
          )}
        </div>

        <p className="text-center text-gray-600 mt-4">
          실시간으로 YOLO 객체 검출 + Segformer 세그멘테이션 결과가 표시됩니다.
        </p>
      </div>

      <div className="mb-md w-full max-w-4xl flex justify-center gap-6">
        {/* 카메라 선택 콤보박스 */}
        <select
          className="border border-gray-300 rounded-lg px-4 py-2 text-gray-700 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={selectedCamera}
          onChange={(e) => {
            setSelectedCamera(e.target.value); // 인덱스를 서버에 전달
            setReloadKey((prev) => prev + 1); // 선택 변경 시 새로고침
          }}
        >
          {cameras.length === 0 ? (
            <option>카메라 없음</option>
          ) : (
            cameras.map((cam, i) => (
              <option key={cam.deviceId} value={cam.deviceId}>
                {cam.label || `Camera ${i}`}
              </option>
            ))
          )}
        </select>

        {/* 시작/중지 버튼 */}
        <button
          onClick={handleToggleStream}
          className={`px-4 py-2 rounded-lg text-white transition ${
            streaming ? "bg-red-500 hover:bg-red-600" : "bg-green-500 hover:bg-green-600"
          }`}
        >
          {streaming ? "중지" : "시작"}
        </button>

        <button
          onClick={handleViewLogs}
          className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg transition"
        >기록 보기</button>

        <button
          onClick={handleSaveCSV}
          className="bg-orange-500 hover:bg-orange-600 text-white px-4 py-2 rounded-lg transition"
        >기록 저장</button>
      </div>

      {/* 기록 표시 */}
      {showLogs && (
        <div className="w-full max-w-4xl overflow-x-auto mt-4">
          <table className="table-auto border-collapse border border-gray-300 w-full text-center">
            <thead>
              <tr>
                <th className="border border-gray-300 px-2 py-1">시간</th>
                <th className="border border-gray-300 px-2 py-1">경고</th>
                <th className="border border-gray-300 px-2 py-1">전체 사람</th>
                <th className="border border-gray-300 px-2 py-1">오버랩</th>
                <th className="border border-gray-300 px-2 py-1">오버랩 아님</th>
              </tr>
            </thead>
            <tbody>
              {logs.map((log, i) => (
                <tr key={i} className="even:bg-gray-50">
                  <td className="border border-gray-300 px-2 py-1">{log.timestamp}</td>
                  <td className="border border-gray-300 px-2 py-1">{log.alert ? "⚠️" : "✅"}</td>
                  <td className="border border-gray-300 px-2 py-1">{log.total_people}</td>
                  <td className="border border-gray-300 px-2 py-1">{log.overlap_count}</td>
                  <td className="border border-gray-300 px-2 py-1">{log.non_overlap_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

    </div>
  );
}