import * as fs from "fs";
import * as path from "path";
import * as semver from "semver";

// Plugin interfaces
export interface PluginMetadata {
  name: string;
  version: string;
  description: string;
  author?: string;
  license?: string;
  homepage?: string;
  repository?: string;
  keywords?: string[];
  engines?: {
    coordinator?: string;
    node?: string;
  };
}

export interface PluginHooks {
  // Lifecycle hooks
  onLoad?: (context: PluginContext) => Promise<void>;
  onUnload?: (context: PluginContext) => Promise<void>;

  // Task hooks
  onTaskCreated?: (task: any, context: PluginContext) => Promise<void>;
  onTaskClaimed?: (task: any, context: PluginContext) => Promise<void>;
  onTaskStarted?: (task: any, context: PluginContext) => Promise<void>;
  onTaskCompleted?: (task: any, context: PluginContext) => Promise<void>;
  onTaskFailed?: (task: any, context: PluginContext) => Promise<void>;

  // Worker hooks
  onWorkerRegistered?: (worker: any, context: PluginContext) => Promise<void>;
  onWorkerHeartbeat?: (worker: any, context: PluginContext) => Promise<void>;
  onWorkerDisconnected?: (worker: any, context: PluginContext) => Promise<void>;

  // Discovery hooks
  onDiscoveryAdded?: (discovery: any, context: PluginContext) => Promise<void>;

  // Coordination hooks
  onCoordinationStart?: (goal: string, context: PluginContext) => Promise<void>;
  onCoordinationComplete?: (context: PluginContext) => Promise<void>;
}

export interface PluginCommands {
  [commandName: string]: {
    description: string;
    options?: {
      flags: string;
      description: string;
      default?: any;
    }[];
    action: (args: any, context: PluginContext) => Promise<void>;
  };
}

export interface Plugin {
  metadata: PluginMetadata;
  hooks?: PluginHooks;
  commands?: PluginCommands;
}

export interface PluginContext {
  coordinationDir: string;
  pluginDir: string;
  config: Record<string, any>;
  logger: PluginLogger;
  api: PluginAPI;
}

export interface PluginLogger {
  info: (message: string) => void;
  warn: (message: string) => void;
  error: (message: string) => void;
  debug: (message: string) => void;
}

export interface PluginAPI {
  // Task operations
  getTasks: () => Promise<any[]>;
  getTask: (id: string) => Promise<any | null>;
  createTask: (data: any) => Promise<any>;
  updateTask: (id: string, data: any) => Promise<any | null>;
  deleteTask: (id: string) => Promise<boolean>;

  // Worker operations
  getWorkers: () => Promise<any[]>;
  getWorker: (id: string) => Promise<any | null>;
  registerWorker: (data: any) => Promise<any>;
  updateWorker: (id: string, data: any) => Promise<any | null>;

  // Discovery operations
  getDiscoveries: () => Promise<any[]>;
  addDiscovery: (data: any) => Promise<any>;

  // Utility
  readConfig: (key: string) => any;
  writeConfig: (key: string, value: any) => void;
}

// Plugin Manager
export class PluginManager {
  private plugins: Map<string, Plugin> = new Map();
  private loadedPlugins: Map<string, PluginContext> = new Map();
  private pluginsDir: string;
  private coordinationDir: string;
  private configPath: string;
  private config: Record<string, any> = {};

  constructor(coordinationDir: string, pluginsDir?: string) {
    this.coordinationDir = coordinationDir;
    this.pluginsDir = pluginsDir || path.join(coordinationDir, "plugins");
    this.configPath = path.join(this.pluginsDir, "config.json");
    this.loadConfig();
  }

  private loadConfig(): void {
    if (fs.existsSync(this.configPath)) {
      try {
        this.config = JSON.parse(fs.readFileSync(this.configPath, "utf-8"));
      } catch {
        this.config = {};
      }
    }
  }

  private saveConfig(): void {
    this.ensureDir(this.pluginsDir);
    fs.writeFileSync(this.configPath, JSON.stringify(this.config, null, 2));
  }

  private ensureDir(dirPath: string): void {
    if (!fs.existsSync(dirPath)) {
      fs.mkdirSync(dirPath, { recursive: true });
    }
  }

  // Create logger for plugin
  private createLogger(pluginName: string): PluginLogger {
    const prefix = `[${pluginName}]`;
    return {
      info: (msg: string) => console.log(`${prefix} INFO: ${msg}`),
      warn: (msg: string) => console.warn(`${prefix} WARN: ${msg}`),
      error: (msg: string) => console.error(`${prefix} ERROR: ${msg}`),
      debug: (msg: string) => {
        if (process.env.DEBUG) {
          console.debug(`${prefix} DEBUG: ${msg}`);
        }
      },
    };
  }

  // Create API for plugin
  private createAPI(pluginName: string): PluginAPI {
    const tasksPath = path.join(this.coordinationDir, "tasks.json");
    const workersPath = path.join(this.coordinationDir, "workers.json");
    const discoveriesPath = path.join(this.coordinationDir, "discoveries.json");

    const readJson = (filePath: string): any => {
      if (!fs.existsSync(filePath)) return null;
      try {
        return JSON.parse(fs.readFileSync(filePath, "utf-8"));
      } catch {
        return null;
      }
    };

    const writeJson = (filePath: string, data: any): void => {
      this.ensureDir(path.dirname(filePath));
      fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
    };

    return {
      getTasks: async () => readJson(tasksPath)?.tasks || [],
      getTask: async (id: string) => {
        const tasks = readJson(tasksPath)?.tasks || [];
        return tasks.find((t: any) => t.id === id) || null;
      },
      createTask: async (data: any) => {
        const tasksData = readJson(tasksPath) || { tasks: [] };
        const task = {
          id: `task-${Date.now()}`,
          ...data,
          created_at: new Date().toISOString(),
        };
        tasksData.tasks.push(task);
        writeJson(tasksPath, tasksData);
        return task;
      },
      updateTask: async (id: string, data: any) => {
        const tasksData = readJson(tasksPath) || { tasks: [] };
        const index = tasksData.tasks.findIndex((t: any) => t.id === id);
        if (index === -1) return null;
        tasksData.tasks[index] = {
          ...tasksData.tasks[index],
          ...data,
          updated_at: new Date().toISOString(),
        };
        writeJson(tasksPath, tasksData);
        return tasksData.tasks[index];
      },
      deleteTask: async (id: string) => {
        const tasksData = readJson(tasksPath) || { tasks: [] };
        const index = tasksData.tasks.findIndex((t: any) => t.id === id);
        if (index === -1) return false;
        tasksData.tasks.splice(index, 1);
        writeJson(tasksPath, tasksData);
        return true;
      },

      getWorkers: async () => readJson(workersPath)?.workers || [],
      getWorker: async (id: string) => {
        const workers = readJson(workersPath)?.workers || [];
        return workers.find((w: any) => w.id === id) || null;
      },
      registerWorker: async (data: any) => {
        const workersData = readJson(workersPath) || { workers: [] };
        const worker = {
          id: `worker-${Date.now()}`,
          ...data,
          last_heartbeat: new Date().toISOString(),
        };
        workersData.workers.push(worker);
        writeJson(workersPath, workersData);
        return worker;
      },
      updateWorker: async (id: string, data: any) => {
        const workersData = readJson(workersPath) || { workers: [] };
        const index = workersData.workers.findIndex((w: any) => w.id === id);
        if (index === -1) return null;
        workersData.workers[index] = { ...workersData.workers[index], ...data };
        writeJson(workersPath, workersData);
        return workersData.workers[index];
      },

      getDiscoveries: async () => readJson(discoveriesPath)?.discoveries || [],
      addDiscovery: async (data: any) => {
        const discoveriesData = readJson(discoveriesPath) || {
          discoveries: [],
        };
        const discovery = {
          id: `discovery-${Date.now()}`,
          ...data,
          created_at: new Date().toISOString(),
        };
        discoveriesData.discoveries.push(discovery);
        writeJson(discoveriesPath, discoveriesData);
        return discovery;
      },

      readConfig: (key: string) => this.config[pluginName]?.[key],
      writeConfig: (key: string, value: any) => {
        if (!this.config[pluginName]) {
          this.config[pluginName] = {};
        }
        this.config[pluginName][key] = value;
        this.saveConfig();
      },
    };
  }

  // Load a plugin from directory
  async loadPlugin(pluginPath: string): Promise<Plugin | null> {
    try {
      const packageJsonPath = path.join(pluginPath, "package.json");
      const indexPath = path.join(pluginPath, "index.js");

      if (!fs.existsSync(packageJsonPath)) {
        console.error(`Plugin package.json not found: ${packageJsonPath}`);
        return null;
      }

      const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, "utf-8"));

      // Check version compatibility
      if (packageJson.engines?.coordinator) {
        const coordinatorVersion = "1.0.0"; // Current coordinator version
        if (
          !semver.satisfies(coordinatorVersion, packageJson.engines.coordinator)
        ) {
          console.error(
            `Plugin ${packageJson.name} requires coordinator version ${packageJson.engines.coordinator}`,
          );
          return null;
        }
      }

      // Load plugin module
      if (!fs.existsSync(indexPath)) {
        console.error(`Plugin entry point not found: ${indexPath}`);
        return null;
      }

      const pluginModule = require(indexPath);
      const plugin: Plugin = {
        metadata: {
          name: packageJson.name,
          version: packageJson.version,
          description: packageJson.description || "",
          author: packageJson.author,
          license: packageJson.license,
          homepage: packageJson.homepage,
          repository: packageJson.repository?.url || packageJson.repository,
          keywords: packageJson.keywords,
          engines: packageJson.engines,
        },
        hooks: pluginModule.hooks,
        commands: pluginModule.commands,
      };

      this.plugins.set(plugin.metadata.name, plugin);

      // Create context and call onLoad
      const context: PluginContext = {
        coordinationDir: this.coordinationDir,
        pluginDir: pluginPath,
        config: this.config[plugin.metadata.name] || {},
        logger: this.createLogger(plugin.metadata.name),
        api: this.createAPI(plugin.metadata.name),
      };

      this.loadedPlugins.set(plugin.metadata.name, context);

      if (plugin.hooks?.onLoad) {
        await plugin.hooks.onLoad(context);
      }

      return plugin;
    } catch (error) {
      console.error(`Failed to load plugin from ${pluginPath}:`, error);
      return null;
    }
  }

  // Unload a plugin
  async unloadPlugin(name: string): Promise<boolean> {
    const plugin = this.plugins.get(name);
    const context = this.loadedPlugins.get(name);

    if (!plugin || !context) {
      return false;
    }

    if (plugin.hooks?.onUnload) {
      await plugin.hooks.onUnload(context);
    }

    this.plugins.delete(name);
    this.loadedPlugins.delete(name);
    return true;
  }

  // Load all plugins from plugins directory
  async loadAllPlugins(): Promise<void> {
    if (!fs.existsSync(this.pluginsDir)) {
      return;
    }

    const entries = fs.readdirSync(this.pluginsDir, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.isDirectory() && !entry.name.startsWith(".")) {
        const pluginPath = path.join(this.pluginsDir, entry.name);
        await this.loadPlugin(pluginPath);
      }
    }
  }

  // Get all loaded plugins
  getPlugins(): Plugin[] {
    return Array.from(this.plugins.values());
  }

  // Get plugin by name
  getPlugin(name: string): Plugin | undefined {
    return this.plugins.get(name);
  }

  // Get all commands from all plugins
  getAllCommands(): Map<string, { plugin: string; command: any }> {
    const commands = new Map<string, { plugin: string; command: any }>();

    for (const [pluginName, plugin] of this.plugins) {
      if (plugin.commands) {
        for (const [cmdName, cmd] of Object.entries(plugin.commands)) {
          commands.set(`${pluginName}:${cmdName}`, {
            plugin: pluginName,
            command: cmd,
          });
        }
      }
    }

    return commands;
  }

  // Execute a hook for all plugins
  async executeHook(
    hookName: keyof PluginHooks,
    ...args: any[]
  ): Promise<void> {
    for (const [pluginName, plugin] of this.plugins) {
      const hook = plugin.hooks?.[hookName];
      if (hook && typeof hook === "function") {
        const context = this.loadedPlugins.get(pluginName);
        if (context) {
          try {
            await (hook as any)(...args, context);
          } catch (error) {
            console.error(
              `Error in plugin ${pluginName} hook ${hookName}:`,
              error,
            );
          }
        }
      }
    }
  }

  // Execute a command
  async executeCommand(commandPath: string, args: any): Promise<void> {
    const [pluginName, cmdName] = commandPath.split(":");
    const plugin = this.plugins.get(pluginName);
    const context = this.loadedPlugins.get(pluginName);

    if (!plugin || !context) {
      throw new Error(`Plugin not found: ${pluginName}`);
    }

    const command = plugin.commands?.[cmdName];
    if (!command) {
      throw new Error(`Command not found: ${cmdName}`);
    }

    await command.action(args, context);
  }

  // Install a plugin from npm or local path
  async installPlugin(source: string): Promise<boolean> {
    this.ensureDir(this.pluginsDir);

    // If source is a local path
    if (fs.existsSync(source)) {
      const packageJson = JSON.parse(
        fs.readFileSync(path.join(source, "package.json"), "utf-8"),
      );
      const destPath = path.join(this.pluginsDir, packageJson.name);

      // Copy plugin to plugins directory
      this.copyDir(source, destPath);
      await this.loadPlugin(destPath);
      return true;
    }

    // For npm packages, you would use npm install programmatically
    // This is a simplified version
    console.log(`Installing ${source} from npm is not implemented yet.`);
    console.log("Please install manually and place in the plugins directory.");
    return false;
  }

  private copyDir(src: string, dest: string): void {
    this.ensureDir(dest);
    const entries = fs.readdirSync(src, { withFileTypes: true });

    for (const entry of entries) {
      const srcPath = path.join(src, entry.name);
      const destPath = path.join(dest, entry.name);

      if (entry.isDirectory()) {
        this.copyDir(srcPath, destPath);
      } else {
        fs.copyFileSync(srcPath, destPath);
      }
    }
  }

  // Uninstall a plugin
  async uninstallPlugin(name: string): Promise<boolean> {
    await this.unloadPlugin(name);

    const pluginPath = path.join(this.pluginsDir, name);
    if (fs.existsSync(pluginPath)) {
      fs.rmSync(pluginPath, { recursive: true });
      return true;
    }

    return false;
  }
}

export default PluginManager;
