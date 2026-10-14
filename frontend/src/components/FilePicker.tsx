import { useMemo, useState } from 'react'

interface Node { name: string; path: string; children: Map<string, Node>; file: boolean }

function build(docs: string[]): Node {
  const root: Node = { name: '', path: '', children: new Map(), file: false }
  for (const doc of docs) {
    let node = root
    doc.split('/').forEach((part, i, parts) => {
      const path = parts.slice(0, i + 1).join('/')
      if (!node.children.has(part)) node.children.set(part, { name: part, path, children: new Map(), file: i === parts.length - 1 })
      node = node.children.get(part)!
    })
  }
  return root
}
const filesUnder = (n: Node): string[] => (n.file ? [n.path] : [...n.children.values()].flatMap(filesUnder))

function Row({ node, selected, toggle, filter, depth }: { node: Node; selected: Set<string>; toggle: (files: string[], on: boolean) => void; filter: string; depth: number }) {
  const [open, setOpen] = useState(true)
  const files = filesUnder(node)
  const visible = !filter || files.some(f => f.toLowerCase().includes(filter))
  if (!visible) return null
  const count = files.filter(f => selected.has(f)).length
  const all = count === files.length, some = count > 0 && !all
  const children = [...node.children.values()].sort((a, b) => Number(a.file) - Number(b.file) || a.name.localeCompare(b.name, undefined, { numeric: true }))
  return (
    <div style={{ marginLeft: depth ? 16 : 0 }}>
      <label className="node" style={{ cursor: 'pointer' }} title={node.path}>
        {node.file ? <span className="caret" /> : <span className="caret" onClick={e => { e.preventDefault(); setOpen(!open) }}>{open ? '▾' : '▸'}</span>}
        <input type="checkbox" checked={all} ref={el => { if (el) el.indeterminate = some }} onChange={e => toggle(files, e.target.checked)} />
        <span style={{ fontWeight: node.file ? 400 : 600 }}>{node.name}</span>
        {!node.file && <span className="muted small">{count}/{files.length}</span>}
      </label>
      {!node.file && open && children.map(c => <Row key={c.path} node={c} selected={selected} toggle={toggle} filter={filter} depth={depth + 1} />)}
    </div>
  )
}

export function FilePicker({ docs, selected, onChange }: { docs: string[]; selected: Set<string>; onChange: (next: Set<string>) => void }) {
  const [filter, setFilter] = useState('')
  const tree = useMemo(() => build(docs), [docs])
  const toggle = (files: string[], on: boolean) => { const next = new Set(selected); files.forEach(f => (on ? next.add(f) : next.delete(f))); onChange(next) }
  return (
    <div className="picker">
      <div className="row" style={{ margin: '4px 0 0' }}>
        <label className="search">Filter files<input value={filter} onChange={e => setFilter(e.target.value.toLowerCase())} placeholder="e.g. ch5, syllabus, pptx" /></label>
        <button onClick={() => onChange(new Set(docs.filter(d => !filter || d.toLowerCase().includes(filter))))}>Select {filter ? 'matching' : 'all'}</button>
        <button onClick={() => onChange(new Set())}>Clear</button>
      </div>
      <div className="tree">{[...tree.children.values()].map(n => <Row key={n.path} node={n} selected={selected} toggle={toggle} filter={filter} depth={0} />)}</div>
    </div>
  )
}
