/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import * as fs from 'fs';
import * as path from 'path';
import { BaseLlmClient } from '../../core/baseLlmClient.js';
import { type Config } from '../../config/config.js';

export interface VectorEntry {
  id: string;
  text: string;
  metadata: Record<string, unknown>;
  embedding: number[];
}

/**
 * A simple vector store with cosine similarity search and disk persistence.
 * This acts as the "Long Term Memory" for the agent.
 */
export class VectorStoreService {
  private entries: VectorEntry[] = [];
  private baseLlmClient?: BaseLlmClient;
  private readonly storagePath: string;

  constructor(private readonly config: Config) {
    this.storagePath = path.join(this.config.storage.getProjectTempDir(), 'vector_store.json');
    this.loadFromDisk();
  }

  private loadFromDisk(): void {
    try {
      if (fs.existsSync(this.storagePath)) {
        const data = fs.readFileSync(this.storagePath, 'utf8');
        this.entries = JSON.parse(data);
      }
    } catch (e) {
      this.entries = [];
    }
  }

  private async saveToDisk(): Promise<void> {
    try {
      const dir = path.dirname(this.storagePath);
      if (!fs.existsSync(dir)) {
        await fs.promises.mkdir(dir, { recursive: true });
      }
      await fs.promises.writeFile(this.storagePath, JSON.stringify(this.entries, null, 2));
    } catch (e) {
      console.error('[RAG] Failed to save to disk:', e);
    }
  }

  private getLlmClient(): BaseLlmClient | undefined {
    if (this.baseLlmClient) return this.baseLlmClient;
    try {
      const contentGenerator = this.config.getContentGenerator();
      if (contentGenerator) {
        this.baseLlmClient = new BaseLlmClient(contentGenerator, this.config);
        return this.baseLlmClient;
      }
    } catch (e) {}
    return undefined;
  }

  /**
   * Adds text to the vector store.
   * Handles chunking and embedding generation.
   */
  async addText(text: string, metadata: Record<string, unknown>): Promise<void> {
    if (!text || text.trim().length === 0) {
      return;
    }

    const llmClient = this.getLlmClient();
    if (!llmClient) return;

    const chunks = this.chunkText(text, 1000);
    const embeddings = await llmClient.generateEmbedding(chunks);

    for (let i = 0; i < chunks.length; i++) {
      this.entries.push({
        id: `${metadata['path'] ?? 'memory'}-${Date.now()}-${i}`,
        text: chunks[i],
        metadata: { ...metadata, chunk_index: i },
        embedding: embeddings[i],
      });
    }
    this.saveToDisk().catch(() => {});
  }

  /**
   * Searches for the most relevant entries based on a query.
   */
  async search(query: string, limit: number = 5): Promise<VectorEntry[]> {
    if (this.entries.length === 0) return [];

    const llmClient = this.getLlmClient();
    if (!llmClient) return [];

    const [queryEmbedding] = await llmClient.generateEmbedding([query]);
    
    const results = this.entries.map(entry => ({
      entry,
      similarity: this.cosineSimilarity(queryEmbedding, entry.embedding)
    }));

    results.sort((a, b) => b.similarity - a.similarity);
    return results.slice(0, limit).map(r => r.entry);
  }

  private chunkText(text: string, maxChars: number): string[] {
    const chunks: string[] = [];
    let current = text;
    while (current.length > 0) {
      if (current.length <= maxChars) {
        chunks.push(current);
        break;
      }
      let splitIndex = current.lastIndexOf('\n\n', maxChars);
      if (splitIndex === -1) splitIndex = current.lastIndexOf('\n', maxChars);
      if (splitIndex === -1) splitIndex = maxChars;
      
      chunks.push(current.substring(0, splitIndex).trim());
      current = current.substring(splitIndex).trim();
    }
    return chunks;
  }

  private cosineSimilarity(vecA: number[], vecB: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < Math.min(vecA.length, vecB.length); i++) {
      dotProduct += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }
    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  /**
   * Clears memory.
   */
  clear(): void {
    this.entries = [];
    if (fs.existsSync(this.storagePath)) {
      fs.unlinkSync(this.storagePath);
    }
  }
}