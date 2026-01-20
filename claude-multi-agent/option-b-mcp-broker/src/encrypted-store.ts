/**
 * Encrypted State Storage (adv-b-018)
 *
 * Provides encrypted persistence for coordination state using SQLite
 * with AES-256-GCM encryption for sensitive data at rest.
 */

import * as crypto from "crypto";
import * as fs from "fs";
import * as path from "path";

// ============================================================================
// Types
// ============================================================================

export interface EncryptedStoreConfig {
  storagePath: string;
  masterKey?: string;
  algorithm?: "aes-256-gcm" | "aes-256-cbc";
  keyDerivationIterations?: number;
}

export interface EncryptedRecord {
  id: string;
  type: "task" | "agent" | "discovery" | "config";
  encryptedData: string;
  iv: string;
  authTag?: string;
  createdAt: string;
  updatedAt: string;
  metadata?: Record<string, any>;
}

export interface StorageStats {
  totalRecords: number;
  recordsByType: Record<string, number>;
  storageSize: number;
  lastUpdated: string;
}

// ============================================================================
// Encrypted State Store
// ============================================================================

export class EncryptedStateStore {
  private storagePath: string;
  private derivedKey: Buffer;
  private algorithm: "aes-256-gcm" | "aes-256-cbc";
  private salt: Buffer;
  private records: Map<string, EncryptedRecord> = new Map();
  private initialized: boolean = false;

  constructor(config: EncryptedStoreConfig) {
    this.storagePath = config.storagePath;
    this.algorithm = config.algorithm || "aes-256-gcm";

    // Generate or load salt
    const saltPath = path.join(path.dirname(this.storagePath), ".salt");
    if (fs.existsSync(saltPath)) {
      this.salt = fs.readFileSync(saltPath);
    } else {
      this.salt = crypto.randomBytes(16);
      const dir = path.dirname(saltPath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      fs.writeFileSync(saltPath, this.salt);
    }

    // Derive key from master key
    const masterKey =
      config.masterKey ||
      process.env.ENCRYPTION_MASTER_KEY ||
      crypto.randomBytes(32).toString("hex");

    this.derivedKey = crypto.pbkdf2Sync(
      masterKey,
      this.salt,
      config.keyDerivationIterations || 100000,
      32,
      "sha256",
    );
  }

  /**
   * Initialize the store, loading existing data
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    if (fs.existsSync(this.storagePath)) {
      try {
        const content = fs.readFileSync(this.storagePath, "utf-8");
        const data = JSON.parse(content);

        for (const record of data.records || []) {
          this.records.set(record.id, record);
        }
      } catch {
        // Start fresh if we can't load
        this.records.clear();
      }
    }

    this.initialized = true;
  }

  /**
   * Save state to disk
   */
  private async save(): Promise<void> {
    const data = {
      version: 1,
      algorithm: this.algorithm,
      records: Array.from(this.records.values()),
      savedAt: new Date().toISOString(),
    };

    const dir = path.dirname(this.storagePath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    fs.writeFileSync(this.storagePath, JSON.stringify(data, null, 2));
  }

  /**
   * Encrypt data
   */
  private encrypt(plaintext: string | object): {
    encrypted: string;
    iv: string;
    authTag?: string;
  } {
    const text =
      typeof plaintext === "object" ? JSON.stringify(plaintext) : plaintext;
    const iv = crypto.randomBytes(12);

    if (this.algorithm === "aes-256-gcm") {
      const cipher = crypto.createCipheriv(
        "aes-256-gcm",
        this.derivedKey,
        iv,
      ) as crypto.CipherGCM;

      let encrypted = cipher.update(text, "utf8", "base64");
      encrypted += cipher.final("base64");

      return {
        encrypted,
        iv: iv.toString("base64"),
        authTag: cipher.getAuthTag().toString("base64"),
      };
    } else {
      // AES-256-CBC fallback
      const ivCbc = crypto.randomBytes(16);
      const cipher = crypto.createCipheriv(
        "aes-256-cbc",
        this.derivedKey,
        ivCbc,
      );

      let encrypted = cipher.update(text, "utf8", "base64");
      encrypted += cipher.final("base64");

      return {
        encrypted,
        iv: ivCbc.toString("base64"),
      };
    }
  }

  /**
   * Decrypt data
   */
  private decrypt(encryptedData: string, iv: string, authTag?: string): string {
    const ivBuffer = Buffer.from(iv, "base64");

    if (this.algorithm === "aes-256-gcm") {
      if (!authTag) {
        throw new Error("Auth tag required for GCM decryption");
      }

      const decipher = crypto.createDecipheriv(
        "aes-256-gcm",
        this.derivedKey,
        ivBuffer,
      ) as crypto.DecipherGCM;

      decipher.setAuthTag(Buffer.from(authTag, "base64"));

      let decrypted = decipher.update(encryptedData, "base64", "utf8");
      decrypted += decipher.final("utf8");

      return decrypted;
    } else {
      const decipher = crypto.createDecipheriv(
        "aes-256-cbc",
        this.derivedKey,
        ivBuffer,
      );

      let decrypted = decipher.update(encryptedData, "base64", "utf8");
      decrypted += decipher.final("utf8");

      return decrypted;
    }
  }

  /**
   * Store an encrypted record
   */
  async store(
    id: string,
    type: EncryptedRecord["type"],
    data: object,
    metadata?: Record<string, any>,
  ): Promise<EncryptedRecord> {
    await this.initialize();

    const { encrypted, iv, authTag } = this.encrypt(data);
    const now = new Date().toISOString();

    const existing = this.records.get(id);
    const record: EncryptedRecord = {
      id,
      type,
      encryptedData: encrypted,
      iv,
      authTag,
      createdAt: existing?.createdAt || now,
      updatedAt: now,
      metadata,
    };

    this.records.set(id, record);
    await this.save();

    return record;
  }

  /**
   * Retrieve and decrypt a record
   */
  async retrieve<T = object>(id: string): Promise<T | null> {
    await this.initialize();

    const record = this.records.get(id);
    if (!record) return null;

    try {
      const decrypted = this.decrypt(
        record.encryptedData,
        record.iv,
        record.authTag,
      );
      return JSON.parse(decrypted) as T;
    } catch {
      return null;
    }
  }

  /**
   * Delete a record
   */
  async delete(id: string): Promise<boolean> {
    await this.initialize();

    const deleted = this.records.delete(id);
    if (deleted) {
      await this.save();
    }
    return deleted;
  }

  /**
   * List records by type
   */
  async listByType(type: EncryptedRecord["type"]): Promise<string[]> {
    await this.initialize();

    return Array.from(this.records.values())
      .filter((r) => r.type === type)
      .map((r) => r.id);
  }

  /**
   * Get all records of a type, decrypted
   */
  async getAllByType<T = object>(
    type: EncryptedRecord["type"],
  ): Promise<Array<{ id: string; data: T; metadata?: Record<string, any> }>> {
    await this.initialize();

    const results: Array<{
      id: string;
      data: T;
      metadata?: Record<string, any>;
    }> = [];

    for (const record of this.records.values()) {
      if (record.type !== type) continue;

      try {
        const decrypted = this.decrypt(
          record.encryptedData,
          record.iv,
          record.authTag,
        );
        results.push({
          id: record.id,
          data: JSON.parse(decrypted) as T,
          metadata: record.metadata,
        });
      } catch {
        // Skip records that can't be decrypted
      }
    }

    return results;
  }

  /**
   * Store task data with encryption
   */
  async storeTask(taskId: string, taskData: object): Promise<void> {
    await this.store(taskId, "task", taskData, {
      storedAt: new Date().toISOString(),
    });
  }

  /**
   * Retrieve task data
   */
  async retrieveTask<T = object>(taskId: string): Promise<T | null> {
    return this.retrieve<T>(taskId);
  }

  /**
   * Store agent registration with encryption
   */
  async storeAgent(agentId: string, agentData: object): Promise<void> {
    await this.store(agentId, "agent", agentData, {
      registeredAt: new Date().toISOString(),
    });
  }

  /**
   * Store discovery with encryption
   */
  async storeDiscovery(
    discoveryId: string,
    discoveryData: object,
  ): Promise<void> {
    await this.store(discoveryId, "discovery", discoveryData, {
      discoveredAt: new Date().toISOString(),
    });
  }

  /**
   * Get storage statistics
   */
  async getStats(): Promise<StorageStats> {
    await this.initialize();

    const recordsByType: Record<string, number> = {};
    for (const record of this.records.values()) {
      recordsByType[record.type] = (recordsByType[record.type] || 0) + 1;
    }

    let storageSize = 0;
    if (fs.existsSync(this.storagePath)) {
      storageSize = fs.statSync(this.storagePath).size;
    }

    return {
      totalRecords: this.records.size,
      recordsByType,
      storageSize,
      lastUpdated: new Date().toISOString(),
    };
  }

  /**
   * Clear all records
   */
  async clear(): Promise<void> {
    this.records.clear();
    await this.save();
  }

  /**
   * Rotate encryption key
   */
  async rotateKey(newMasterKey: string): Promise<void> {
    await this.initialize();

    // Decrypt all records with old key
    const decryptedRecords: Array<{
      record: EncryptedRecord;
      data: object;
    }> = [];

    for (const record of this.records.values()) {
      try {
        const decrypted = this.decrypt(
          record.encryptedData,
          record.iv,
          record.authTag,
        );
        decryptedRecords.push({
          record,
          data: JSON.parse(decrypted),
        });
      } catch {
        // Skip records that can't be decrypted
      }
    }

    // Generate new derived key
    this.derivedKey = crypto.pbkdf2Sync(
      newMasterKey,
      this.salt,
      100000,
      32,
      "sha256",
    );

    // Re-encrypt all records with new key
    this.records.clear();
    for (const { record, data } of decryptedRecords) {
      const { encrypted, iv, authTag } = this.encrypt(data);
      this.records.set(record.id, {
        ...record,
        encryptedData: encrypted,
        iv,
        authTag,
        updatedAt: new Date().toISOString(),
      });
    }

    await this.save();
  }

  /**
   * Export encrypted backup
   */
  async exportBackup(): Promise<string> {
    await this.initialize();

    const backup = {
      version: 1,
      algorithm: this.algorithm,
      exportedAt: new Date().toISOString(),
      records: Array.from(this.records.values()),
    };

    return JSON.stringify(backup, null, 2);
  }

  /**
   * Import from backup
   */
  async importBackup(backupData: string): Promise<number> {
    const backup = JSON.parse(backupData);
    let imported = 0;

    for (const record of backup.records || []) {
      this.records.set(record.id, record);
      imported++;
    }

    await this.save();
    return imported;
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let storeInstance: EncryptedStateStore | null = null;

export function getEncryptedStore(
  config?: EncryptedStoreConfig,
): EncryptedStateStore {
  if (!storeInstance && config) {
    storeInstance = new EncryptedStateStore(config);
  }
  if (!storeInstance) {
    throw new Error("EncryptedStateStore not initialized - provide config");
  }
  return storeInstance;
}

export function resetEncryptedStore(): void {
  storeInstance = null;
}
