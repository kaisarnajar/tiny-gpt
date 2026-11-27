import { useEffect, useState, type FormEvent } from 'react'
import './App.css'

type Message = { role: 'user' | 'assistant'; text: string; error?: boolean }
type Layer = { name: string; shape: string; params: number }

export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [layers, setLayers] = useState<Layer[]>([])
  const [total, setTotal] = useState(0)
  const [showParams, setShowParams] = useState(true)
  const [temperature, setTemperature] = useState(1.0)
  const [topK, setTopK] = useState(0)
  const [topP, setTopP] = useState(0.9)

  useEffect(() => {
    fetch('/api/info')
      .then((res) => res.json())
      .then((data) => {
        setLayers(data.layers)
        setTotal(data.total)
      })
  }, [])

  async function send(e: FormEvent) {
    e.preventDefault()
    const prompt = input.trim()
    if (!prompt || loading) return

    setMessages((m) => [...m, { role: 'user', text: prompt }])
    setInput('')
    setLoading(true)
    try {
      const res = await fetch('/api/complete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, temperature, top_k: topK, top_p: topP }),
      })
      const data = await res.json()
      const reply = data.error ? { text: data.error, error: true } : { text: data.sentence }
      setMessages((m) => [...m, { role: 'assistant', ...reply }])
    } catch {
      setMessages((m) => [...m, { role: 'assistant', text: 'Could not reach the server.', error: true }])
    }
    setLoading(false)
  }

  const greedy = temperature === 0

  return (
    <div className="chat">
      <header className="chat-header">
        <h1>tiny-gpt</h1>
        <span className="chat-header-note">Step 4d · feed-forward network</span>
        <button type="button" className="params-toggle" onClick={() => setShowParams(!showParams)}>
          {total.toLocaleString('en-US')} parameters {showParams ? '▴' : '▾'}
        </button>
      </header>

      {showParams && (
        <section className="params">
          <table>
            <thead>
              <tr>
                <th>Layer</th>
                <th>Shape</th>
                <th className="num">Parameters</th>
              </tr>
            </thead>
            <tbody>
              {layers.map((l) => (
                <tr key={l.name}>
                  <td>{l.name}</td>
                  <td>{l.shape}</td>
                  <td className="num">{l.params.toLocaleString('en-US')}</td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr>
                <td colSpan={2}>Total</td>
                <td className="num">{total.toLocaleString('en-US')}</td>
              </tr>
            </tfoot>
          </table>
        </section>
      )}

      <main className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <p>Type the start of a sentence and the model finishes it.</p>
            <p className="chat-empty-example">Try "The Dog" or "my mother"</p>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`message message-${m.role}${m.error ? ' message-error' : ''}`}>
            {m.text}
          </div>
        ))}
        {loading && <div className="message message-assistant message-typing">…</div>}
      </main>

      <section className="sampling">
        <label>
          <span>Temperature</span>
          <input type="range" min={0} max={2} step={0.1} value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))} />
          <output>{greedy ? '0 · greedy' : temperature.toFixed(1)}</output>
        </label>
        <label className={greedy ? 'off' : ''}>
          <span>Top-k</span>
          <input type="range" min={0} max={20} step={1} value={topK} disabled={greedy}
            onChange={(e) => setTopK(Number(e.target.value))} />
          <output>{topK === 0 ? 'off' : topK}</output>
        </label>
        <label className={greedy ? 'off' : ''}>
          <span>Top-p</span>
          <input type="range" min={0.05} max={1} step={0.05} value={topP} disabled={greedy}
            onChange={(e) => setTopP(Number(e.target.value))} />
          <output>{topP >= 1 ? 'off' : topP.toFixed(2)}</output>
        </label>
      </section>

      <form className="chat-composer" onSubmit={send}>
        <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Message tiny-gpt" autoFocus />
        <button type="submit" disabled={loading || !input.trim()}>Send</button>
      </form>
    </div>
  )
}
