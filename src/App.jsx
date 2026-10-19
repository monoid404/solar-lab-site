import React, { useState, useEffect, useMemo, useRef } from 'react';
import { 
  LineChart, Line, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer, BarChart, Bar 
} from 'recharts';
import { 
  UploadCloud, Activity, Cpu, Target, Database, FileText, 
  Users, AlertCircle, CheckCircle2, ChevronRight, Zap
} from 'lucide-react';

// ----------------------------------------------------------------------
// 1. Math & ML Utilities (vanilla JS implementation to replace scikit-learn)
// ----------------------------------------------------------------------

// 행렬 전치
const transpose = (matrix) => matrix[0].map((col, c) => matrix.map((row, r) => matrix[r][c]));

// 행렬 곱셈
const multiply = (a, b) => {
  const aNumRows = a.length, aNumCols = a[0].length, bNumCols = b[0].length;
  const m = new Array(aNumRows);
  for (let r = 0; r < aNumRows; ++r) {
    m[r] = new Array(bNumCols);
    for (let c = 0; c < bNumCols; ++c) {
      m[r][c] = 0;
      for (let i = 0; i < aNumCols; ++i) {
        m[r][c] += a[r][i] * b[i][c];
      }
    }
  }
  return m;
};

// 가우스-조던 소거법을 이용한 역행렬 계산
const invertMatrix = (matrix) => {
  const n = matrix.length;
  const a = matrix.map(row => [...row]);
  const i = Array.from({ length: n }, (_, row) => Array.from({ length: n }, (_, col) => (row === col ? 1 : 0)));

  for (let p = 0; p < n; p++) {
    let pivot = a[p][p];
    if (pivot === 0) {
      // Find a non-zero pivot
      let swapRow = p + 1;
      while (swapRow < n && a[swapRow][p] === 0) swapRow++;
      if (swapRow === n) return null; // Singular matrix
      let temp = a[p]; a[p] = a[swapRow]; a[swapRow] = temp;
      temp = i[p]; i[p] = i[swapRow]; i[swapRow] = temp;
      pivot = a[p][p];
    }
    for (let c = 0; c < n; c++) {
      a[p][c] /= pivot;
      i[p][c] /= pivot;
    }
    for (let r = 0; r < n; r++) {
      if (r !== p) {
        let factor = a[r][p];
        for (let c = 0; c < n; c++) {
          a[r][c] -= factor * a[p][c];
          i[r][c] -= factor * i[p][c];
        }
      }
    }
  }
  return i;
};

// 다중 선형 회귀 학습 (정규 방정식: Beta = (X^T * X)^-1 * X^T * Y)
const trainLinearRegression = (X, Y) => {
  const X_design = X.map(row => [1, ...row]); // Bias항(1) 추가
  const Y_mat = Y.map(val => [val]);
  const X_T = transpose(X_design);
  const X_T_X = multiply(X_T, X_design);
  const inv_X_T_X = invertMatrix(X_T_X);
  if (!inv_X_T_X) return null; // 역행렬이 존재하지 않음
  const X_T_Y = multiply(X_T, Y_mat);
  const beta = multiply(inv_X_T_X, X_T_Y).map(row => row[0]);
  return beta; // [bias, w1, w2, w3...]
};

// 예측 함수
const predict = (weights, x) => {
  let y_pred = weights[0]; // bias
  for (let i = 0; i < x.length; i++) {
    y_pred += weights[i + 1] * x[i];
  }
  return y_pred;
};

// 통계 유틸리티
const calculateMetrics = (y_true, y_pred) => {
  const n = y_true.length;
  let sse = 0, sst = 0, sum_y = 0, sum_abs_err = 0;
  for (let i = 0; i < n; i++) sum_y += y_true[i];
  const mean_y = sum_y / n;
  
  for (let i = 0; i < n; i++) {
    const err = y_true[i] - y_pred[i];
    sse += err * err;
    sst += (y_true[i] - mean_y) * (y_true[i] - mean_y);
    sum_abs_err += Math.abs(err);
  }
  
  const mae = sum_abs_err / n;
  const mse = sse / n;
  const rmse = Math.sqrt(mse);
  const r2 = sst === 0 ? 0 : 1 - (sse / sst);
  
  return { mae, rmse, r2 };
};

// ----------------------------------------------------------------------
// 2. Main Application Component
// ----------------------------------------------------------------------

export default function SolarLabApp() {
  const [activeTab, setActiveTab] = useState('overview');
  
  // Data State
  const [rawDataset, setRawDataset] = useState(null);
  const [dataStats, setDataStats] = useState(null);
  const [modelData, setModelData] = useState(null);
  const [error, setError] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);

  // Prediction Form State
  const [predInputs, setPredInputs] = useState({ temp: 30, humid: 50, light: 50 });
  const [predResult, setPredResult] = useState(null);

  // ----------------------------------------------------------------------
  // 3. File Processing & AI Pipeline
  // ----------------------------------------------------------------------
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setIsProcessing(true);
    setError('');

    const reader = new FileReader();
    // Use CP949 or EUC-KR for Korean CSV files if UTF-8 fails, but browser FileReader usually defaults to UTF-8.
    // To handle Korean Windows CSV, we try euc-kr.
    reader.readAsText(file, 'euc-kr');
    
    reader.onload = (e) => {
      try {
        const text = e.target.result;
        const lines = text.split('\n').filter(line => line.trim() !== '');
        
        if (lines.length < 2) throw new Error("데이터가 부족하거나 파일 형식이 잘못되었습니다.");
        
        // Parse CSV
        const headers = lines[0].split(',').map(h => h.trim());
        
        // Validate headers based on requirements
        const requiredHeaders = ['온도', '습도', '밝기', '태양광 패널 출력 전압'];
        const headerIndices = {};
        requiredHeaders.forEach(reqH => {
          const idx = headers.findIndex(h => h.includes(reqH));
          if (idx === -1) throw new Error(`필수 열을 찾을 수 없습니다: ${reqH}`);
          headerIndices[reqH] = idx;
        });

        const parsedData = [];
        for (let i = 1; i < lines.length; i++) {
          const values = lines[i].split(',');
          if (values.length >= headers.length) {
            const row = {
              temp: parseFloat(values[headerIndices['온도']]),
              humid: parseFloat(values[headerIndices['습도']]),
              light: parseFloat(values[headerIndices['밝기']]),
              voltage: parseFloat(values[headerIndices['태양광 패널 출력 전압']])
            };
            if (!isNaN(row.temp) && !isNaN(row.humid) && !isNaN(row.light) && !isNaN(row.voltage)) {
              parsedData.push(row);
            }
          }
        }

        if (parsedData.length === 0) throw new Error("유효한 데이터 행이 없습니다.");

        // Calculate Stats
        const stats = {
          count: parsedData.length,
          temp: { min: Math.min(...parsedData.map(d=>d.temp)), max: Math.max(...parsedData.map(d=>d.temp)), mean: parsedData.reduce((s,d)=>s+d.temp,0)/parsedData.length },
          humid: { min: Math.min(...parsedData.map(d=>d.humid)), max: Math.max(...parsedData.map(d=>d.humid)), mean: parsedData.reduce((s,d)=>s+d.humid,0)/parsedData.length },
          light: { min: Math.min(...parsedData.map(d=>d.light)), max: Math.max(...parsedData.map(d=>d.light)), mean: parsedData.reduce((s,d)=>s+d.light,0)/parsedData.length },
        };

        // Split Data 80/20
        const trainSize = Math.floor(parsedData.length * 0.8);
        // Basic shuffle (pseudo-random) for train/test split
        const shuffled = [...parsedData].sort(() => 0.5 - Math.random()); 
        const trainData = shuffled.slice(0, trainSize);
        const testData = shuffled.slice(trainSize);

        const X_train = trainData.map(d => [d.temp, d.humid, d.light]);
        const y_train = trainData.map(d => d.voltage);
        const X_test = testData.map(d => [d.temp, d.humid, d.light]);
        const y_test = testData.map(d => d.voltage);

        // Train Model
        const weights = trainLinearRegression(X_train, y_train);
        if (!weights) throw new Error("모델 학습 실패 (역행렬 계산 불가)");

        // Evaluate on Test Data
        const y_pred = X_test.map(x => predict(weights, x));
        const metrics = calculateMetrics(y_test, y_pred);

        // Set States
        setRawDataset(parsedData);
        setDataStats(stats);
        setModelData({
          weights,
          metrics,
          trainSize,
          testSize: testData.length
        });
        
        // Init predict inputs to mean values
        setPredInputs({
          temp: Math.round(stats.temp.mean),
          humid: Math.round(stats.humid.mean),
          light: Math.round(stats.light.mean)
        });

        setIsProcessing(false);
      } catch (err) {
        setError(err.message);
        setIsProcessing(false);
      }
    };
    reader.onerror = () => {
      setError("파일을 읽는 중 오류가 발생했습니다.");
      setIsProcessing(false);
    };
  };

  const handlePredict = () => {
    if (!modelData) return;
    setIsProcessing(true);
    // Simulate slight delay for UI feedback
    setTimeout(() => {
      const { temp, humid, light } = predInputs;
      const res = predict(modelData.weights, [temp, humid, light]);
      setPredResult(res);
      setIsProcessing(false);
    }, 400);
  };

  // ----------------------------------------------------------------------
  // 4. UI Components
  // ----------------------------------------------------------------------
  
  const navItems = [
    { id: 'overview', icon: <Activity size={18}/>, label: '1. Overview' },
    { id: 'data', icon: <Database size={18}/>, label: '2. Data Intelligence' },
    { id: 'predict', icon: <Target size={18}/>, label: '3. Prediction Lab' },
    { id: 'model', icon: <Cpu size={18}/>, label: '4. Model Insight' },
    { id: 'journey', icon: <FileText size={18}/>, label: '5. Research Journey' },
    { id: 'about', icon: <Users size={18}/>, label: '6. About Green Narae' },
  ];

  const MetricCard = ({ title, value, icon, subtitle }) => (
    <div className="bg-[#161b22] border border-[#30363d] rounded-xl p-5 flex flex-col">
      <div className="flex items-center gap-3 text-[#8b949e] mb-2">
        {icon}
        <span className="font-semibold">{title}</span>
      </div>
      <div className="text-3xl font-bold text-white mb-1">{value}</div>
      {subtitle && <div className="text-sm text-[#58a6ff]">{subtitle}</div>}
    </div>
  );

  return (
    <div className="flex h-screen bg-[#0d1117] text-[#c9d1d9] font-sans overflow-hidden selection:bg-[#58a6ff] selection:text-white">
      
      {/* Sidebar */}
      <div className="w-72 bg-[#161b22] border-r border-[#30363d] flex flex-col shadow-2xl z-10">
        <div className="p-6">
          <h1 className="text-2xl font-black text-transparent bg-clip-text bg-gradient-to-r from-[#58a6ff] to-[#3fb950] tracking-tight">
            SOLAR LAB
          </h1>
          <p className="text-sm text-[#8b949e] mt-1 font-medium">GREEN NARAE | 계산여고</p>
        </div>
        
        <nav className="flex-1 px-4 py-2 space-y-2 overflow-y-auto">
          {navItems.map(item => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm transition-all duration-200 ${
                activeTab === item.id 
                  ? 'bg-gradient-to-r from-[#1f6feb]/20 to-transparent text-[#58a6ff] border border-[#1f6feb]/30 shadow-[inset_4px_0_0_#58a6ff]' 
                  : 'text-[#8b949e] hover:bg-[#21262d] hover:text-white'
              }`}
            >
              {item.icon}
              <span className="font-medium tracking-wide">{item.label}</span>
            </button>
          ))}
        </nav>
        
        <div className="p-6 border-t border-[#30363d]">
          <div className="text-xs text-[#8b949e] flex items-center justify-between">
            <span>2026 Joint Science</span>
            <Zap size={14} className="text-[#d29922]" />
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden relative">
        
        {/* Header Header */}
        <header className="h-16 flex items-center justify-between px-8 bg-[#0d1117]/80 backdrop-blur-md border-b border-[#30363d] z-10">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            {navItems.find(i => i.id === activeTab)?.label}
          </h2>
          <div className="flex items-center gap-4 text-sm font-medium">
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border ${rawDataset ? 'bg-[#2ea043]/10 text-[#3fb950] border-[#2ea043]/30' : 'bg-[#da3633]/10 text-[#ff7b72] border-[#da3633]/30'}`}>
              <Database size={14} />
              {rawDataset ? `Data: ${rawDataset.length.toLocaleString()} rows` : 'Data: 연결 필요'}
            </div>
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border ${modelData ? 'bg-[#388bfd]/10 text-[#58a6ff] border-[#388bfd]/30' : 'bg-[#da3633]/10 text-[#ff7b72] border-[#da3633]/30'}`}>
              <Cpu size={14} />
              {modelData ? 'AI Model: Active' : 'AI Model: Offline'}
            </div>
          </div>
        </header>

        {/* Global File Upload Overlay (if no data) */}
        {!rawDataset && activeTab !== 'about' && activeTab !== 'journey' && (
          <div className="absolute inset-0 z-50 bg-[#0d1117]/90 backdrop-blur-sm flex flex-col items-center justify-center p-8">
            <div className="bg-[#161b22] border border-[#30363d] rounded-2xl p-10 max-w-lg w-full text-center shadow-2xl">
              <div className="w-20 h-20 bg-gradient-to-br from-[#1f6feb] to-[#238636] rounded-full flex items-center justify-center mx-auto mb-6 shadow-[0_0_30px_rgba(88,166,255,0.3)]">
                <UploadCloud size={40} className="text-white" />
              </div>
              <h2 className="text-2xl font-bold text-white mb-3">데이터 파일 연결</h2>
              <p className="text-[#8b949e] mb-8 leading-relaxed text-sm">
                AI 모델 학습과 시각화를 위해 기준 파일<br/>
                <strong className="text-white bg-[#21262d] px-2 py-1 rounded mx-1">K-EE 분석 데이터.csv</strong>를 업로드해주세요.
              </p>
              
              <label className="relative inline-flex items-center justify-center px-8 py-3.5 text-sm font-bold text-white bg-[#238636] hover:bg-[#2ea043] border border-[rgba(240,246,252,0.1)] rounded-lg cursor-pointer transition-colors w-full group">
                <UploadCloud size={18} className="mr-2 group-hover:scale-110 transition-transform" />
                <span>데이터 파일 선택 (.csv)</span>
                <input 
                  type="file" 
                  accept=".csv" 
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  onChange={handleFileUpload}
                  disabled={isProcessing}
                />
              </label>
              
              {isProcessing && <p className="mt-4 text-[#58a6ff] text-sm animate-pulse flex items-center justify-center gap-2"><Activity size={14}/> 데이터 분석 및 AI 모델 학습 중...</p>}
              {error && <p className="mt-4 text-[#ff7b72] text-sm flex items-center justify-center gap-1"><AlertCircle size={14}/> {error}</p>}
            </div>
          </div>
        )}

        {/* Scrollable Content */}
        <main className="flex-1 overflow-y-auto p-8 lg:p-12 pb-24 relative">
          <div className="max-w-6xl mx-auto">
            
            {/* --- TAB 1: OVERVIEW --- */}
            {activeTab === 'overview' && (
              <div className="space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-700">
                
                {/* Hero */}
                <div className="text-center space-y-6 py-12 relative">
                  <div className="absolute inset-0 bg-gradient-to-b from-[#1f6feb]/5 to-transparent blur-3xl -z-10 rounded-full"></div>
                  <h1 className="text-5xl lg:text-7xl font-black text-white tracking-tighter">
                    환경을 읽고,<br/>
                    <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#d29922] to-[#e3b341]">태양광의 출력을 예측하다.</span>
                  </h1>
                  <p className="text-lg text-[#8b949e] max-w-2xl mx-auto font-medium leading-relaxed">
                    환경 데이터 기반 태양광 패널 출력 전압 예측 인공지능 모델.<br/>
                    다양한 환경 요인과 회귀 분석 알고리즘을 활용해 출력 전압을 예측하고, 서로 다른 환경에서의 활용 가능성을 탐구합니다.
                  </p>
                </div>

                {/* Core Question */}
                <div className="bg-gradient-to-r from-[#1f6feb]/10 to-[#238636]/10 border border-[#1f6feb]/20 rounded-2xl p-8 relative overflow-hidden">
                  <div className="absolute top-0 right-0 p-8 opacity-10"><Target size={120} /></div>
                  <h3 className="text-xl font-bold text-white mb-2 flex items-center gap-2">
                    <AlertCircle className="text-[#58a6ff]" /> 핵심 연구 질문
                  </h3>
                  <p className="text-2xl font-bold text-white leading-tight mb-4">
                    "우리 학교에서 학습한 AI는 다른 학교에서도 정확할까?"
                  </p>
                  <p className="text-[#8b949e] max-w-3xl">
                    학습 데이터와 평가 데이터를 철저히 분리해 예측 오차를 분석하고, 서로 다른 기후 및 설치 환경에서도 
                    우리 모델이 범용적으로 활용될 수 있는지 검증하는 것을 연구의 최종 목표로 삼습니다. (타 학교 검증은 향후 연구 계획 포함)
                  </p>
                </div>

                {/* Metrics */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <MetricCard title="전체 데이터 수" value={rawDataset ? rawDataset.length.toLocaleString() : "-"} subtitle="rows" icon={<Database size={20} className="text-[#58a6ff]" />} />
                  <MetricCard title="환경 변수 (Features)" value="3" subtitle="온도, 습도, 밝기" icon={<Activity size={20} className="text-[#d29922]" />} />
                  <MetricCard title="적용 모델" value={modelData ? "Linear Regression" : "-"} subtitle="다중 선형 회귀 알고리즘" icon={<Cpu size={20} className="text-[#bc8cff]" />} />
                  <MetricCard title="모델 성능 (R²)" value={modelData ? modelData.metrics.r2.toFixed(3) : "-"} subtitle="내부 Test Set 검증 완료" icon={<CheckCircle2 size={20} className="text-[#3fb950]" />} />
                </div>

                {/* Pipeline */}
                <div>
                  <h3 className="text-xl font-bold text-white mb-6">Research Pipeline</h3>
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    {[
                      { step: '01', title: 'DATA', desc: '환경 센서 및 태양광 전압 데이터 수집', status: '완료', color: 'from-[#2ea043]/20 to-transparent border-[#2ea043]/30' },
                      { step: '02', title: 'ANALYSIS', desc: '결측치 제거 및 상관관계 통계 분석', status: '완료', color: 'from-[#388bfd]/20 to-transparent border-[#388bfd]/30' },
                      { step: '03', title: 'LEARNING', desc: '다중 선형 회귀 모델 학습 (80% Data)', status: '완료', color: 'from-[#bc8cff]/20 to-transparent border-[#bc8cff]/30' },
                      { step: '04', title: 'VALIDATION', desc: 'Test Set 성능 평가 및 타 학교 적용 확장', status: '진행 중', color: 'from-[#d29922]/20 to-transparent border-[#d29922]/30' },
                    ].map((s, i) => (
                      <div key={i} className={`bg-gradient-to-b ${s.color} border rounded-xl p-5 relative overflow-hidden`}>
                        <div className="text-4xl font-black text-white/5 absolute -right-2 -bottom-4">{s.step}</div>
                        <div className="text-xs font-bold text-[#8b949e] mb-1">STEP {s.step}</div>
                        <h4 className="text-lg font-bold text-white mb-2">{s.title}</h4>
                        <p className="text-sm text-[#8b949e] mb-4 h-10">{s.desc}</p>
                        <span className={`text-xs font-bold px-2 py-1 rounded bg-[#21262d] ${s.status === '완료' ? 'text-[#3fb950]' : 'text-[#d29922]'}`}>
                          {s.status}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* --- TAB 2: DATA INTELLIGENCE --- */}
            {activeTab === 'data' && rawDataset && (
              <div className="space-y-8 animate-in fade-in duration-500">
                <div>
                  <h2 className="text-3xl font-bold text-white mb-2">Data Intelligence</h2>
                  <p className="text-[#8b949e]">수집된 기초 데이터의 구조적 특성과 변수 간의 관계를 시각적으로 분석합니다.</p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  {/* Basic Stats */}
                  <div className="lg:col-span-1 bg-[#161b22] border border-[#30363d] rounded-xl p-6">
                    <h3 className="text-lg font-bold text-white mb-4">기초 통계 (Summary)</h3>
                    <div className="space-y-4">
                      {[
                        { name: '온도 (°C)', stat: dataStats.temp, color: 'text-[#ff7b72]' },
                        { name: '습도 (%)', stat: dataStats.humid, color: 'text-[#58a6ff]' },
                        { name: '밝기', stat: dataStats.light, color: 'text-[#d29922]' },
                      ].map((item, i) => (
                        <div key={i} className="bg-[#0d1117] rounded-lg p-3 border border-[#30363d]">
                          <div className={`text-sm font-bold mb-2 ${item.color}`}>{item.name}</div>
                          <div className="flex justify-between text-xs text-[#8b949e]">
                            <span>최소: <strong className="text-white">{item.stat.min}</strong></span>
                            <span>평균: <strong className="text-white">{item.stat.mean.toFixed(2)}</strong></span>
                            <span>최대: <strong className="text-white">{item.stat.max}</strong></span>
                          </div>
                        </div>
                      ))}
                    </div>
                    <div className="mt-6 p-3 bg-[#da3633]/10 border border-[#da3633]/20 rounded-lg text-sm text-[#ff7b72] flex gap-2">
                      <AlertCircle size={16} className="shrink-0 mt-0.5" />
                      <div>
                        <strong>해석 주의 원칙</strong><br/>
                        데이터의 분산 분석 및 산점도는 두 변수 간의 통계적 <i>상관관계</i>를 나타낼 뿐, 직접적인 <i>인과관계</i>를 단정하지 않습니다.
                      </div>
                    </div>
                  </div>

                  {/* Scatter Plots */}
                  <div className="lg:col-span-2 bg-[#161b22] border border-[#30363d] rounded-xl p-6">
                    <h3 className="text-lg font-bold text-white mb-4">환경 변수에 따른 출력 전압 산점도 (Scatter Plot)</h3>
                    <p className="text-sm text-[#8b949e] mb-6">데이터 샘플링(최대 500개)을 통해 전체적인 분포 경향을 시각화합니다.</p>
                    
                    <div className="h-72 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                          <XAxis type="number" dataKey="light" name="밝기" stroke="#8b949e" label={{ value: '밝기', position: 'insideBottom', offset: -10, fill: '#8b949e' }} />
                          <YAxis type="number" dataKey="voltage" name="출력 전압" stroke="#8b949e" label={{ value: '출력 전압 (V)', angle: -90, position: 'insideLeft', fill: '#8b949e' }} />
                          <Tooltip 
                            cursor={{ strokeDasharray: '3 3' }}
                            contentStyle={{ backgroundColor: '#161b22', borderColor: '#30363d', color: '#c9d1d9' }}
                          />
                          <Scatter name="Data" data={rawDataset.slice(0, 500)} fill="#d29922" opacity={0.6} />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* --- TAB 3: PREDICTION LAB --- */}
            {activeTab === 'predict' && modelData && (
              <div className="space-y-8 animate-in fade-in duration-500">
                <div>
                  <h2 className="text-3xl font-bold text-white mb-2">Prediction Lab</h2>
                  <p className="text-[#8b949e]">학습된 인공지능 모델을 이용해 임의의 환경 조건에서 패널의 출력 전압을 시뮬레이션합니다.</p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  
                  {/* Inputs */}
                  <div className="bg-[#161b22] border border-[#30363d] rounded-xl p-8 shadow-lg">
                    <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                      <Activity className="text-[#58a6ff]" /> 측정 환경 입력
                    </h3>
                    
                    <div className="space-y-8">
                      {/* Temp Slider */}
                      <div>
                        <div className="flex justify-between mb-2">
                          <label className="text-sm font-bold text-[#c9d1d9]">온도 (°C)</label>
                          <span className="text-sm text-[#58a6ff] font-mono bg-[#0d1117] px-2 py-0.5 rounded">{predInputs.temp}</span>
                        </div>
                        <input 
                          type="range" min="0" max="60" step="0.1" 
                          value={predInputs.temp} 
                          onChange={(e) => setPredInputs({...predInputs, temp: parseFloat(e.target.value)})}
                          className="w-full accent-[#58a6ff]" 
                        />
                        <div className="flex justify-between text-xs text-[#8b949e] mt-1">
                          <span>학습 데이터 최소: {dataStats.temp.min}°C</span>
                          <span>최대: {dataStats.temp.max}°C</span>
                        </div>
                      </div>

                      {/* Humid Slider */}
                      <div>
                        <div className="flex justify-between mb-2">
                          <label className="text-sm font-bold text-[#c9d1d9]">습도 (%)</label>
                          <span className="text-sm text-[#3fb950] font-mono bg-[#0d1117] px-2 py-0.5 rounded">{predInputs.humid}</span>
                        </div>
                        <input 
                          type="range" min="0" max="100" step="1" 
                          value={predInputs.humid} 
                          onChange={(e) => setPredInputs({...predInputs, humid: parseFloat(e.target.value)})}
                          className="w-full accent-[#3fb950]" 
                        />
                      </div>

                      {/* Light Slider */}
                      <div>
                        <div className="flex justify-between mb-2">
                          <label className="text-sm font-bold text-[#c9d1d9]">밝기</label>
                          <span className="text-sm text-[#d29922] font-mono bg-[#0d1117] px-2 py-0.5 rounded">{predInputs.light}</span>
                        </div>
                        <input 
                          type="range" min={Math.max(0, dataStats.light.min - 50)} max={dataStats.light.max + 50} step="1" 
                          value={predInputs.light} 
                          onChange={(e) => setPredInputs({...predInputs, light: parseFloat(e.target.value)})}
                          className="w-full accent-[#d29922]" 
                        />
                      </div>
                    </div>

                    <button 
                      onClick={handlePredict}
                      disabled={isProcessing}
                      className="w-full mt-10 bg-gradient-to-r from-[#1f6feb] to-[#238636] hover:from-[#1f6feb]/90 hover:to-[#238636]/90 text-white font-bold py-4 rounded-lg flex items-center justify-center gap-2 transition-all shadow-[0_0_20px_rgba(31,111,235,0.3)]"
                    >
                      {isProcessing ? <Activity className="animate-spin" /> : <Target />}
                      출력 전압 예측 실행
                    </button>
                  </div>

                  {/* Result */}
                  <div className="flex flex-col gap-6">
                    <div className="bg-[#161b22] border border-[#30363d] rounded-xl p-8 shadow-lg flex-1 flex flex-col items-center justify-center relative overflow-hidden">
                      <div className="absolute top-0 right-0 p-4 opacity-5"><Zap size={100}/></div>
                      <h3 className="text-lg font-bold text-[#8b949e] mb-4">AI 예측 출력 전압</h3>
                      
                      {predResult !== null ? (
                        <div className="text-center animate-in zoom-in duration-300">
                          <div className="text-6xl lg:text-7xl font-black text-white mb-2 tracking-tighter">
                            {predResult.toFixed(4)} <span className="text-3xl text-[#58a6ff]">V</span>
                          </div>
                          <p className="text-sm text-[#8b949e]">
                            입력된 {predInputs.temp}°C, {predInputs.humid}%, 밝기 {predInputs.light} 조건에서의 회귀 예측값입니다.
                          </p>
                        </div>
                      ) : (
                        <div className="text-[#8b949e] text-center italic">
                          좌측 패널에서 환경 조건을 설정한 후<br/>'예측 실행' 버튼을 클릭하세요.
                        </div>
                      )}
                    </div>

                    {/* Scientific Integrity Warning */}
                    <div className="bg-[#d29922]/10 border border-[#d29922]/30 rounded-xl p-5 flex items-start gap-4">
                      <AlertCircle className="text-[#d29922] shrink-0 mt-0.5" />
                      <div className="text-sm text-[#e3b341]">
                        <strong className="block mb-1">과학적 근거 제한 안내 (Scientific Integrity)</strong>
                        현재 수집된 데이터에는 전압(V)만 존재하며, 저항 부하 또는 전류(A) 데이터가 측정되지 않았습니다. 
                        따라서 임의의 가정을 통한 발전 전력(W = V × I)이나 에너지(Wh) 산출은 모델의 신뢰성을 저해하므로 
                        본 플랫폼에서는 <b>계산 및 표기하지 않습니다.</b> (향후 전류 측정 실험을 통해 확장 예정)
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* --- TAB 4: MODEL INSIGHT --- */}
            {activeTab === 'model' && modelData && (
              <div className="space-y-8 animate-in fade-in duration-500">
                <div>
                  <h2 className="text-3xl font-bold text-white mb-2">Model Insight</h2>
                  <p className="text-[#8b949e]">학습된 인공지능 모델의 내부 구조, 가중치(Weights) 및 검증 지표를 투명하게 공개합니다.</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                  <div className="bg-[#161b22] border border-[#30363d] rounded-xl p-5">
                    <div className="text-sm text-[#8b949e] mb-1">R-Squared (결정계수)</div>
                    <div className="text-3xl font-bold text-[#3fb950]">{modelData.metrics.r2.toFixed(4)}</div>
                    <div className="text-xs text-[#8b949e] mt-2">1에 가까울수록 높은 설명력을 의미</div>
                  </div>
                  <div className="bg-[#161b22] border border-[#30363d] rounded-xl p-5">
                    <div className="text-sm text-[#8b949e] mb-1">MAE (평균 절대 오차)</div>
                    <div className="text-3xl font-bold text-[#58a6ff]">{modelData.metrics.mae.toFixed(4)}</div>
                    <div className="text-xs text-[#8b949e] mt-2">실제값과 예측값 차이의 평균</div>
                  </div>
                  <div className="bg-[#161b22] border border-[#30363d] rounded-xl p-5">
                    <div className="text-sm text-[#8b949e] mb-1">RMSE (평균 제곱근 오차)</div>
                    <div className="text-3xl font-bold text-[#bc8cff]">{modelData.metrics.rmse.toFixed(4)}</div>
                    <div className="text-xs text-[#8b949e] mt-2">큰 오차에 더 가중치를 둔 평가 지표</div>
                  </div>
                </div>

                <div className="bg-[#161b22] border border-[#30363d] rounded-xl p-8">
                  <h3 className="text-xl font-bold text-white mb-4">선형 회귀 다항식 (Linear Regression Equation)</h3>
                  <p className="text-[#8b949e] mb-4 text-sm">
                    Orange3 워크플로를 재현하여 학습된 모델의 가중치 배열은 다음과 같습니다. 데이터 분할은 학습 80%({modelData.trainSize}행), 검증 20%({modelData.testSize}행)를 엄격히 준수하였습니다.
                  </p>
                  
                  <div className="bg-[#0d1117] p-6 rounded-lg font-mono text-sm lg:text-base border border-[#30363d] overflow-x-auto whitespace-nowrap">
                    <span className="text-[#c9d1d9]">Output Voltage (V) = </span>
                    <br/><span className="text-[#58a6ff] ml-4">{modelData.weights[0].toFixed(5)}</span> <span className="text-[#8b949e] text-xs">(Bias)</span>
                    <br/><span className="text-white ml-4">+ ( </span><span className="text-[#ff7b72]">{modelData.weights[1].toFixed(5)}</span><span className="text-white"> × 온도 )</span>
                    <br/><span className="text-white ml-4">+ ( </span><span className="text-[#3fb950]">{modelData.weights[2].toFixed(5)}</span><span className="text-white"> × 습도 )</span>
                    <br/><span className="text-white ml-4">+ ( </span><span className="text-[#d29922]">{modelData.weights[3].toFixed(5)}</span><span className="text-white"> × 밝기 )</span>
                  </div>
                </div>
              </div>
            )}

            {/* --- TAB 5: RESEARCH JOURNEY --- */}
            {activeTab === 'journey' && (
              <div className="space-y-8 animate-in fade-in duration-500 max-w-4xl mx-auto">
                <div className="text-center mb-12">
                  <h2 className="text-3xl font-bold text-white mb-4">Research Journey</h2>
                  <p className="text-[#8b949e]">그린나래 팀의 치열한 탐구 과정과 향후 계획을 소개합니다.</p>
                </div>

                <div className="relative border-l border-[#30363d] ml-4 space-y-12 pb-8">
                  {[
                    { title: "탐구 동기 및 연구 질문 설정", status: "Done", desc: "태양광 에너지 효율에 미치는 요인을 파악하고, 단일 환경을 넘어 다양한 환경에서도 적응 가능한 범용적 인공지능 예측 모델의 필요성을 인식하여 연구를 시작했습니다." },
                    { title: "환경 데이터 수집 및 정제", status: "Done", desc: "교내에 센서를 설치하여 온도, 습도, 밝기, 출력 전압 데이터를 8,400여 건 수집했습니다. 결측값과 이상값을 제거하여 학습에 적합한 데이터셋을 구축했습니다." },
                    { title: "데이터 분석 (Orange3 기반)", status: "Done", desc: "Orange3 소프트웨어를 활용해 데이터의 기초 통계와 상관관계를 분석하고, 다중 선형 회귀 분석 워크플로를 설계했습니다." },
                    { title: "모델 구축 및 내부 성능 평가", status: "Done", desc: "학습(80%)과 평가(20%) 데이터를 분리하여 모델을 학습시켰으며, 미학습 데이터를 이용해 MAE, RMSE, R² 지표로 정확도를 검증했습니다." },
                    { title: "오류 원인 분석 및 한계 검토", status: "In Progress", desc: "예측값이 빗나간 이상치 데이터의 원인을 파악 중이며, 전류 미측정으로 인한 전력(W) 계산의 불가 등 현재 연구의 한계를 명확히 규정했습니다." },
                    { title: "타 학교 데이터 융합 및 확장", status: "Plan", desc: "지능형 과학실 ON 플랫폼을 통해 다른 기후 조건과 환경을 가진 학교의 데이터를 확보하여 모델의 일반화(Generalization) 성능을 최종 검증할 계획입니다." },
                  ].map((step, i) => (
                    <div key={i} className="relative pl-8">
                      <div className={`absolute -left-3.5 top-1 h-7 w-7 rounded-full border-4 border-[#0d1117] flex items-center justify-center
                        ${step.status === 'Done' ? 'bg-[#3fb950]' : step.status === 'In Progress' ? 'bg-[#d29922]' : 'bg-[#30363d]'}`}>
                        {step.status === 'Done' && <CheckCircle2 size={14} className="text-[#0d1117]" />}
                      </div>
                      <h4 className="text-xl font-bold text-white mb-2 flex items-center gap-3">
                        {step.title}
                        <span className={`text-xs px-2 py-0.5 rounded font-medium
                          ${step.status === 'Done' ? 'bg-[#2ea043]/20 text-[#3fb950]' : 
                            step.status === 'In Progress' ? 'bg-[#d29922]/20 text-[#d29922]' : 'bg-[#30363d] text-[#8b949e]'}`}>
                          {step.status}
                        </span>
                      </h4>
                      <p className="text-[#8b949e] leading-relaxed">{step.desc}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* --- TAB 6: ABOUT --- */}
            {activeTab === 'about' && (
              <div className="space-y-12 animate-in fade-in duration-500 max-w-5xl mx-auto">
                <div className="text-center mb-12 py-10 border-b border-[#30363d]">
                  <h1 className="text-4xl font-black text-white mb-2">GREEN NARAE</h1>
                  <p className="text-xl text-[#8b949e]">인천 계산여자고등학교 연구팀</p>
                </div>

                <div>
                  <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
                    <Users className="text-[#58a6ff]" /> Student Researchers
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="bg-[#161b22] border border-[#30363d] rounded-xl p-6 hover:border-[#58a6ff]/50 transition-colors">
                      <h4 className="text-xl font-bold text-white mb-1">한지윤</h4>
                      <div className="text-sm text-[#58a6ff] mb-4">Data Engineer</div>
                      <p className="text-sm text-[#8b949e]">지능형 과학실 ON 데이터 취합 및 정리, 결측값·이상값 확인, AI 학습을 위한 데이터 전처리 담당.</p>
                    </div>
                    <div className="bg-[#161b22] border border-[#30363d] rounded-xl p-6 hover:border-[#3fb950]/50 transition-colors">
                      <h4 className="text-xl font-bold text-white mb-1">황윤아</h4>
                      <div className="text-sm text-[#3fb950] mb-4">Environment Sensing</div>
                      <p className="text-sm text-[#8b949e]">태양광 패널과 환경 센서 설치, 측정 환경 관리, 환경 데이터 및 출력 전압 데이터 수집 관리 담당.</p>
                    </div>
                    <div className="bg-[#161b22] border border-[#30363d] rounded-xl p-6 hover:border-[#bc8cff]/50 transition-colors">
                      <h4 className="text-xl font-bold text-white mb-1">조하율</h4>
                      <div className="text-sm text-[#bc8cff] mb-4">AI Modeling & Viz</div>
                      <p className="text-sm text-[#8b949e]">Orange3 회귀 모델 제작, 모델별 성능 비교, 예측 결과 시각화 및 탐구 결과 정리 담당.</p>
                    </div>
                  </div>
                </div>

                <div className="mt-12 pt-8 border-t border-[#30363d]">
                  <h3 className="text-2xl font-bold text-white mb-6">Faculty Advisor</h3>
                  <div className="bg-gradient-to-r from-[#161b22] to-[#0d1117] border border-[#30363d] rounded-xl p-6 inline-block">
                    <h4 className="text-xl font-bold text-white mb-1">노효성 선생님</h4>
                    <p className="text-sm text-[#8b949e]">실험 장비 활용 지원, 실험 안전 관리, 탐구 결과의 타당성 및 과학적 신뢰성 검토.</p>
                  </div>
                </div>
              </div>
            )}
            
          </div>
        </main>
      </div>
    </div>
  );
}