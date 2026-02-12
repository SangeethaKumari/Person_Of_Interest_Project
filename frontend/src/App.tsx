import React, { useState, useEffect } from 'react';
import { Search, Upload, Zap, Terminal, Eye, Sparkles, History as HistoryIcon, X, Clock, Target, Rocket, Cpu } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

// Helper for tailwind classes
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface SearchResult {
  path: string;
  score: number;
  raw_score: number;
}

interface ModelResults {
  [key: string]: SearchResult[];
}

interface HistoryItem {
  id: string;
  type: 'text' | 'image';
  query: string;
  timestamp: number;
}

const MODELS = [
  { id: 'base_clip', name: 'Base CLIP', icon: Cpu, desc: 'ViT-B-32 (Fast & Lean)', color: 'text-blue-400', bg: 'bg-blue-500/10' },
  { id: 'enhanced_clip_l', name: 'Enhanced CLIP-L', icon: Rocket, desc: 'ViT-L-14 (High Precision)', color: 'text-purple-400', bg: 'bg-purple-500/10' },
  { id: 'siglip2', name: 'SigLIP 2', icon: Target, desc: 'Google (Advanced Reasoning)', color: 'text-emerald-400', bg: 'bg-emerald-500/10' },
];

export default function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<ModelResults | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'text' | 'image' | 'camera'>('text');
  const videoRef = React.useRef<HTMLVideoElement>(null);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [showHistory, setShowHistory] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem('poi_history');
    if (saved) {
      try {
        setHistory(JSON.parse(saved));
      } catch (e) { console.error(e); }
    }
  }, []);

  const addToHistory = (type: 'text' | 'image', query: string) => {
    const newItem: HistoryItem = {
      id: Math.random().toString(36).substr(2, 9),
      type, query, timestamp: Date.now()
    };
    const updated = [newItem, ...history].slice(0, 10);
    setHistory(updated);
    localStorage.setItem('poi_history', JSON.stringify(updated));
  };

  const handleSearch = async (e?: React.FormEvent, customQuery?: string) => {
    if (e) e.preventDefault();
    const searchQuery = customQuery || query;
    if (!searchQuery) return;

    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_BASE_URL}/search/all?query=${encodeURIComponent(searchQuery)}`);
      setResults(response.data);
      addToHistory('text', searchQuery);
      if (customQuery) setQuery(customQuery);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Connection Error');
    } finally {
      setLoading(false);
    }
  };

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setPreviewImage(URL.createObjectURL(file));
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_BASE_URL}/search/image?model_type=all`, formData);
      setResults(response.data);
      setActiveTab('image');
      addToHistory('image', 'Image Analysis');
    } catch (err) {
      alert('Upload failed');
    } finally {
      setLoading(false);
    }
  };

  const startCamera = async () => {
    setActiveTab('camera');
    setPreviewImage(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      alert("Camera access denied");
    }
  };

  const captureImage = () => {
    if (videoRef.current && canvasRef.current) {
      const context = canvasRef.current.getContext('2d');
      if (context) {
        canvasRef.current.width = videoRef.current.videoWidth;
        canvasRef.current.height = videoRef.current.videoHeight;
        context.drawImage(videoRef.current, 0, 0);

        canvasRef.current.toBlob(async (blob) => {
          if (blob) {
            const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
            setPreviewImage(URL.createObjectURL(file));

            // Stop the camera stream
            const stream = videoRef.current?.srcObject as MediaStream;
            stream?.getTracks().forEach(track => track.stop());

            setLoading(true);
            const formData = new FormData();
            formData.append('file', file);

            try {
              const response = await axios.post(`${API_BASE_URL}/search/image?model_type=all`, formData);
              setResults(response.data);
              addToHistory('image', 'Live Camera Scan');
            } catch (err) {
              alert('Analysis failed');
            } finally {
              setLoading(false);
            }
          }
        }, 'image/jpeg');
      }
    }
  };

  return (
    <div className="min-h-screen px-4 py-12 md:px-8 relative overflow-hidden bg-[#020617] text-slate-100">
      {/* Background Decorative Elements */}
      <div className="fixed top-0 left-0 w-full h-full pointer-events-none overflow-hidden -z-10">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-600/10 blur-[120px] rounded-full" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-600/10 blur-[120px] rounded-full" />
      </div>

      {/* History Sidebar */}
      <AnimatePresence>
        {showHistory && (
          <>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={() => setShowHistory(false)} className="fixed inset-0 bg-black/80 backdrop-blur-md z-40" />
            <motion.div initial={{ x: '100%' }} animate={{ x: 0 }} exit={{ x: '100%' }} className="fixed right-0 top-0 h-full w-full max-w-sm bg-slate-900/90 border-l border-slate-800 z-50 p-8 shadow-2xl backdrop-blur-xl">
              <div className="flex items-center justify-between mb-8">
                <div className="flex items-center gap-3 font-bold text-xl"><HistoryIcon className="text-blue-400" /> Intelligence Log</div>
                <button onClick={() => setShowHistory(false)} className="p-2 hover:bg-slate-800 rounded-full transition-colors"><X /></button>
              </div>
              <div className="space-y-4">
                {history.map((item) => (
                  <button key={item.id} onClick={() => { if (item.type === 'text') handleSearch(undefined, item.query); setShowHistory(false); }} className="w-full text-left p-4 rounded-2xl bg-slate-800/30 hover:bg-slate-800/60 transition-all border border-slate-700/50 flex items-center justify-between group">
                    <div className="flex items-center gap-3 overflow-hidden">
                      {item.type === 'image' ? <Upload size={14} className="text-purple-400" /> : <Terminal size={14} className="text-emerald-400" />}
                      <span className="truncate text-sm font-medium">{item.query}</span>
                    </div>
                  </button>
                ))}
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>

      {/* Header */}
      <header className="max-w-6xl mx-auto mb-16 text-center relative">
        <div className="absolute top-0 right-0 flex gap-4">
          <button onClick={() => setShowHistory(true)} className="flex items-center gap-2 px-5 py-2.5 rounded-full bg-slate-800/40 hover:bg-slate-800 border border-slate-700/50 transition-all group backdrop-blur-md">
            <Clock size={16} className="text-blue-400 group-hover:rotate-12 transition-transform" />
            <span className="text-xs font-black uppercase tracking-widest text-slate-300">History</span>
          </button>
        </div>

        <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="flex items-center justify-center gap-4 mb-6">
          <div className="p-3 rounded-2xl bg-gradient-to-br from-blue-500/20 to-purple-500/20 text-blue-400 border border-blue-500/20 shadow-[0_0_30px_rgba(59,130,246,0.2)]">
            <Eye size={40} />
          </div>
          <h1 className="text-5xl md:text-7xl font-black tracking-tighter title-gradient drop-shadow-sm">
            POI TRIPLE-VAL
          </h1>
        </motion.div>
        <p className="text-slate-400 max-w-2xl mx-auto text-xl font-medium">
          Triple-Engine Cross-Evaluation.
          <span className="block mt-2 text-sm text-slate-500 font-mono tracking-widest uppercase text-emerald-400">SigLIP 2 Active: Re-calibrated for high-reasoning queries</span>
        </p>
      </header>

      {/* Models Status View */}
      <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
        {MODELS.map((model) => (
          <div key={model.id} className="p-6 rounded-3xl glass-morphism border border-white/5 flex items-center justify-between group hover:border-blue-500/20 transition-all">
            <div className="space-y-1">
              <span className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500 mb-2 block">{model.name}</span>
              <div className="flex items-center gap-2">
                <model.icon className={cn("size-5", model.color)} />
                <span className="text-2xl font-black tracking-tighter">Online</span>
              </div>
              <p className="text-[10px] font-mono text-slate-500">{model.desc}</p>
            </div>
            <div className={cn("size-12 rounded-2xl flex items-center justify-center", model.bg)}>
              <Zap className={cn("size-6 fill-current", model.color)} />
            </div>
          </div>
        ))}
      </div>

      <main className="max-w-7xl mx-auto space-y-12 pb-24">
        {/* Search Controller */}
        <div className="max-w-4xl mx-auto space-y-8">
          <div className="flex items-center justify-center gap-4 mb-4">
            <button
              onClick={() => setActiveTab('text')}
              className={cn("px-8 py-3 rounded-2xl font-black tracking-tighter transition-all uppercase text-xs", activeTab === 'text' ? "bg-blue-600 text-white shadow-xl shadow-blue-600/20" : "bg-slate-800/40 text-slate-500 hover:text-slate-300")}
            >
              Semantic Query
            </button>
            <button
              onClick={() => setActiveTab('image')}
              className={cn("px-8 py-3 rounded-2xl font-black tracking-tighter transition-all uppercase text-xs", activeTab === 'image' ? "bg-purple-600 text-white shadow-xl shadow-purple-600/20" : "bg-slate-800/40 text-slate-500 hover:text-slate-300")}
            >
              Identity Upload
            </button>
            <button
              onClick={startCamera}
              className={cn("px-8 py-3 rounded-2xl font-black tracking-tighter transition-all uppercase text-xs", activeTab === 'camera' ? "bg-emerald-600 text-white shadow-xl shadow-emerald-600/20" : "bg-slate-800/40 text-slate-500 hover:text-slate-300")}
            >
              Live Scan
            </button>
          </div>

          <div className="glass-morphism p-4 rounded-[40px] border border-white/5 relative">
            {activeTab === 'text' ? (
              <form onSubmit={handleSearch} className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Describe target..."
                  className="flex-1 bg-transparent py-4 px-6 md:py-6 md:px-10 outline-none text-xl md:text-2xl font-bold tracking-tight text-white placeholder:text-slate-700 text-center sm:text-left"
                />
                <button type="submit" disabled={loading} className="h-16 w-full sm:w-20 sm:h-20 rounded-[25px] sm:rounded-[30px] bg-blue-600 flex items-center justify-center hover:bg-blue-500 shadow-xl shadow-blue-600/30 transition-all active:scale-95">
                  {loading ? <Zap className="animate-spin text-white" /> : <Search size={28} className="text-white" />}
                </button>
              </form>
            ) : activeTab === 'image' ? (
              <label className="flex items-center justify-center py-6 px-10 gap-6 cursor-pointer group">
                <input type="file" className="hidden" onChange={handleImageUpload} accept="image/*" />
                <div className="size-20 rounded-[30px] bg-purple-600 flex items-center justify-center group-hover:bg-purple-500 shadow-xl shadow-purple-600/30 transition-all">
                  {loading ? <Sparkles className="animate-bounce text-white" /> : <Upload size={32} className="text-white" />}
                </div>
                <div className="flex-1 text-2xl font-bold text-slate-600 group-hover:text-white transition-colors">
                  {previewImage ? "Scanning Identity..." : "Drop target image for triple lookup"}
                </div>
              </label>
            ) : (
              <div className="flex flex-col items-center justify-center gap-4">
                {!previewImage ? (
                  <>
                    <div className="relative w-full max-w-md aspect-[3/4] rounded-3xl overflow-hidden bg-black border-2 border-slate-700">
                      <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
                      <div className="absolute inset-0 border-[3px] border-emerald-500/50 rounded-3xl pointer-events-none" />
                    </div>
                    <button onClick={captureImage} className="px-8 py-3 rounded-full bg-emerald-600 text-white font-bold flex items-center gap-2 hover:bg-emerald-500 transition-all">
                      <div className="size-4 bg-white rounded-full animate-pulse" /> Capture Identity
                    </button>
                  </>
                ) : (
                  <div className="flex flex-col items-center gap-4">
                    <div className="text-emerald-400 font-mono uppercase tracking-widest animate-pulse">Analyzing Biometrics...</div>
                    <button onClick={() => { setPreviewImage(null); startCamera(); }} className="text-sm text-slate-500 underline hover:text-white">Retake Scan</button>
                  </div>
                )}
                <canvas ref={canvasRef} className="hidden" />
              </div>
            )}
          </div>
          {error && <div className="text-center text-red-400 font-mono text-xs uppercase tracking-widest">{error}</div>}
        </div>

        {/* Results Comparison Grid */}
        <AnimatePresence mode="wait">
          {results && (
            <motion.div initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }} className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              {MODELS.map((model) => (
                <div key={model.id} className="space-y-6">
                  <div className="flex items-center gap-3 border-b border-slate-800 pb-4 mb-4">
                    <model.icon className={cn("size-6", model.color)} />
                    <h3 className="text-xl font-bold tracking-tighter uppercase">{model.name}</h3>
                  </div>

                  <div className="space-y-6">
                    {results[model.id]?.length === 0 ? (
                      <div className="py-20 text-center glass-morphism rounded-3xl text-slate-700 font-mono text-xs tracking-widest uppercase">No Match Identified</div>
                    ) : (
                      results[model.id]?.map((res, i) => (
                        <motion.div key={res.path + model.id} initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * 0.05 }} className="group p-4 rounded-[32px] bg-slate-900/40 border border-white/[0.03] hover:border-blue-500/30 transition-all">
                          <div className="aspect-square rounded-[24px] overflow-hidden relative border border-white/[0.05] shadow-2xl">
                            <img
                              src={`${API_BASE_URL}/static/${res.path}`}
                              className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-700"
                              onError={(e) => { (e.target as HTMLImageElement).src = `https://raw.githubusercontent.com/Almighty-Alpaca/CelebA-dataset/master/images/${res.path.split('/').pop()}`; }}
                            />
                            <div className="absolute top-4 right-4 bg-black/80 backdrop-blur-md px-4 py-2 rounded-2xl border border-white/10">
                              <span className="text-xs font-black text-white italic">{(res.score * 100).toFixed(1)}%</span>
                            </div>
                          </div>
                          <div className="mt-4 px-2">
                            <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                              <motion.div initial={{ width: 0 }} animate={{ width: `${res.score * 100}%` }} className={cn("h-full rounded-full transition-all duration-1000", model.color.replace('text', 'bg'))} />
                            </div>
                            <div className="mt-3 flex items-center justify-between text-[10px] font-mono text-slate-600 tracking-tighter">
                              <span>SIGNATURE</span>
                              <span>{res.path.split('/').pop()}</span>
                            </div>
                          </div>
                        </motion.div>
                      ))
                    )}
                  </div>
                </div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>

      </main>

      {/* Dynamic Grid Overlay */}
      <div className="fixed inset-0 pointer-events-none grid grid-cols-6 grid-rows-6 opacity-[0.03] -z-20">
        {[...Array(36)].map((_, i) => <div key={i} className="border-[0.5px] border-slate-600" />)}
      </div>
    </div>
  );
}
