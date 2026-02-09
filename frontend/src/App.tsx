import React, { useState } from 'react';
import { Search, Upload, User, Zap, Github, Terminal, Info, Eye, Sparkles } from 'lucide-react';
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

const MODELS = [
  { id: 'base_clip', name: 'Base CLIP', desc: 'ViT-B-32 (Fast)' },
  { id: 'enhanced_clip_l', name: 'Enhanced CLIP-L', desc: 'ViT-L-14 (Accurate)' },
  { id: 'siglip2', name: 'SigLIP 2', desc: 'Google (Advanced)' }
];

export default function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<ModelResults | null>(null);
  const [loading, setLoading] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'text' | 'image'>('text');
  const [previewImage, setPreviewImage] = useState<string | null>(null);

  const handleTextSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query) return;

    setLoading(true);
    try {
      const allResults: ModelResults = {};
      await Promise.all(MODELS.map(async (model) => {
        const response = await axios.post(`${API_BASE_URL}/search/text?query=${encodeURIComponent(query)}&model_type=${model.id}`);
        allResults[model.id] = response.data.results;
      }));
      setResults(allResults);
    } catch (err) {
      console.error(err);
      alert('Error searching. Make sure the backend is running at :8000');
    } finally {
      setLoading(false);
    }
  };

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setPreviewImage(URL.createObjectURL(file));
    setUploadLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const allResults: ModelResults = {};
      await Promise.all(MODELS.map(async (model) => {
        const response = await axios.post(`${API_BASE_URL}/search/image?model_type=${model.id}`, formData);
        allResults[model.id] = response.data.results;
      }));
      setResults(allResults);
      setActiveTab('image');
    } catch (err) {
      console.error(err);
      alert('Error uploading image.');
    } finally {
      setUploadLoading(false);
    }
  };

  return (
    <div className="min-h-screen px-4 py-12 md:px-8">
      {/* Header */}
      <header className="max-w-6xl mx-auto mb-16 text-center">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-center gap-2 mb-4"
        >
          <div className="p-2 rounded-xl bg-blue-500/20 text-blue-400">
            <Eye size={32} />
          </div>
          <h1 className="text-4xl md:text-6xl font-black tracking-tighter title-gradient">
            POI SEARCH
          </h1>
        </motion.div>
        <p className="text-slate-400 max-w-2xl mx-auto text-lg">
          An advanced AI-powered person retrieval system using CLIP and SigLIP 2.
          Discover your celebrity doppelgänger or search via natural language.
        </p>
      </header>

      <main className="max-w-7xl mx-auto space-y-12">
        {/* Search Controls */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">

          {/* Text Search Panel */}
          <div className={cn(
            "lg:col-span-7 glass-morphism p-8 rounded-3xl space-y-6 transition-all duration-500",
            activeTab === 'text' ? "ring-2 ring-blue-500/50" : "opacity-60 grayscale hover:opacity-100 hover:grayscale-0"
          )} onClick={() => setActiveTab('text')}>
            <div className="flex items-center gap-3 mb-2">
              <Terminal className="text-emerald-400" />
              <h2 className="text-xl font-bold">Narrative Search</h2>
            </div>
            <form onSubmit={handleTextSearch} className="relative group">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g., A smiling woman with blonde hair and blue eyes..."
                className="w-full bg-slate-900/50 border border-slate-700/50 rounded-2xl py-5 px-6 outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-transparent transition-all pr-16 text-lg"
              />
              <button
                type="submit"
                className="absolute right-3 top-2.5 p-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white transition-all shadow-lg active:scale-95 disabled:opacity-50"
                disabled={loading}
              >
                {loading ? <Zap className="animate-spin" /> : <Search />}
              </button>
            </form>
            <div className="flex flex-wrap gap-2 text-sm">
              <span className="text-slate-500">Suggestions:</span>
              {['Man with beard', 'Person wearing hat', 'Blue eyes'].map(s => (
                <button
                  key={s}
                  onClick={() => setQuery(s)}
                  className="px-3 py-1 rounded-full bg-slate-800 hover:bg-slate-700 transition-colors border border-slate-700"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>

          {/* Image Upload Panel */}
          <div className={cn(
            "lg:col-span-5 glass-morphism p-8 rounded-3xl space-y-6 transition-all duration-500",
            activeTab === 'image' ? "ring-2 ring-purple-500/50" : "opacity-60 grayscale hover:opacity-100 hover:grayscale-0"
          )} onClick={() => setActiveTab('image')}>
            <div className="flex items-center gap-3 mb-2">
              <User className="text-purple-400" />
              <h2 className="text-xl font-bold">Doppelgänger Finder</h2>
            </div>
            <label className="relative flex flex-col items-center justify-center border-2 border-dashed border-slate-700 rounded-2xl p-8 hover:bg-slate-900/50 cursor-pointer transition-all group overflow-hidden h-[130px]">
              <input type="file" className="hidden" onChange={handleImageUpload} accept="image/*" />
              {previewImage ? (
                <img src={previewImage} className="absolute inset-0 w-full h-full object-cover opacity-30 blur-sm" />
              ) : null}
              <div className="relative flex flex-col items-center gap-2">
                {uploadLoading ? <Sparkles className="animate-bounce text-purple-400" /> : <Upload className="group-hover:-translate-y-1 transition-transform" />}
                <span className="font-medium text-slate-400">Click to upload photo</span>
              </div>
            </label>
          </div>
        </div>

        {/* Results Grid */}
        <AnimatePresence>
          {results && (
            <motion.div
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8"
            >
              {MODELS.map((model) => (
                <div key={model.id} className="space-y-6">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-4 mb-6">
                    <div>
                      <h3 className="text-xl font-bold">{model.name}</h3>
                      <p className="text-sm text-slate-500">{model.desc}</p>
                    </div>
                    <div className="bg-slate-800 p-2 rounded-lg">
                      <Terminal size={16} className="text-blue-400" />
                    </div>
                  </div>

                  <div className="space-y-4">
                    {results[model.id]?.map((res, i) => (
                      <motion.div
                        key={res.path}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.1 }}
                        className="group flex gap-4 p-4 rounded-2xl bg-white/5 border border-white/5 hover:bg-white/10 transition-all hover:-translate-y-1"
                      >
                        {/* 
                          Note: In production, res.path should be a full URL.
                          For local, we might need to proxy images from the backend.
                        */}
                        <div className="w-24 h-24 rounded-xl bg-slate-800 overflow-hidden flex-shrink-0 border-2 border-slate-700">
                          <img
                            src={`http://localhost:8000/static/${res.path}`}
                            alt="Result"
                            className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                            onError={(e) => {
                              // Fallback for local files if static serving is set
                              (e.target as HTMLImageElement).src = 'https://via.placeholder.com/150';
                            }}
                          />
                        </div>
                        <div className="flex flex-col justify-center gap-2 flex-grow">
                          <div className="flex items-center justify-between">
                            <span className="text-xs font-bold uppercase tracking-widest text-blue-400">Match Confidence</span>
                            <span className="text-sm font-mono font-bold text-white">{(res.score * 100).toFixed(1)}%</span>
                          </div>
                          <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${res.score * 100}%` }}
                              className="h-full bg-gradient-to-r from-blue-500 to-indigo-500 rounded-full shadow-[0_0_10px_rgba(59,130,246,0.5)]"
                            />
                          </div>
                          <p className="text-[10px] text-slate-500 mt-1 font-mono">{res.path.split('/').pop()}</p>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Footer */}
        <footer className="pt-24 pb-8 flex flex-col items-center gap-4 text-slate-500 border-t border-slate-900 mt-24">
          <div className="flex gap-6">
            <a href="#" className="hover:text-white transition-colors"><Github size={20} /></a>
            <a href="#" className="hover:text-white transition-colors"><Terminal size={20} /></a>
            <a href="#" className="hover:text-white transition-colors"><Info size={20} /></a>
          </div>
          <p className="text-sm">Built with Next.js, CLIP, and Qdrant</p>
          <div className="text-[10px] uppercase tracking-[0.2em] font-bold text-slate-700 flex items-center gap-2">
            Powered by <Zap size={10} /> SupportVectors
          </div>
        </footer>
      </main>
    </div>
  );
}
