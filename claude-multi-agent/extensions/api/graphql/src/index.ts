import { ApolloServer } from "@apollo/server";
import { expressMiddleware } from "@apollo/server/express4";
import { ApolloServerPluginDrainHttpServer } from "@apollo/server/plugin/drainHttpServer";
import { makeExecutableSchema } from "@graphql-tools/schema";
import { WebSocketServer } from "ws";
import { useServer } from "graphql-ws/lib/use/ws";
import { PubSub } from "graphql-subscriptions";
import express from "express";
import { createServer } from "http";
import cors from "cors";
import * as fs from "fs";
import * as path from "path";
import { v4 as uuidv4 } from "uuid";
import { config } from "dotenv";

config();

// PubSub for subscriptions
const pubsub = new PubSub();

// Event names
const TASK_CREATED = "TASK_CREATED";
const TASK_UPDATED = "TASK_UPDATED";
const TASK_DELETED = "TASK_DELETED";
const WORKER_UPDATED = "WORKER_UPDATED";
const DISCOVERY_ADDED = "DISCOVERY_ADDED";

// GraphQL Schema
const typeDefs = `#graphql
  enum TaskStatus {
    available
    claimed
    in_progress
    done
    failed
  }

  enum WorkerStatus {
    idle
    busy
    offline
  }

  type Task {
    id: ID!
    description: String!
    status: TaskStatus!
    priority: Int!
    assigned_to: String
    dependencies: [String!]
    created_at: String
    updated_at: String
    result: String
    error: String
    tags: [String!]
  }

  type Worker {
    id: ID!
    status: WorkerStatus!
    current_task: String
    last_heartbeat: String
    tasks_completed: Int
    capabilities: [String!]
  }

  type Discovery {
    id: ID!
    title: String!
    content: String!
    created_by: String!
    created_at: String!
    tags: [String!]
  }

  type TaskStats {
    total: Int!
    available: Int!
    claimed: Int!
    in_progress: Int!
    done: Int!
    failed: Int!
  }

  type WorkerStats {
    total: Int!
    idle: Int!
    busy: Int!
    offline: Int!
  }

  type CoordinationStatus {
    tasks: TaskStats!
    workers: WorkerStats!
    timestamp: String!
  }

  input CreateTaskInput {
    description: String!
    priority: Int
    dependencies: [String!]
    tags: [String!]
  }

  input UpdateTaskInput {
    description: String
    priority: Int
    status: TaskStatus
    assigned_to: String
    dependencies: [String!]
    tags: [String!]
    result: String
    error: String
  }

  input RegisterWorkerInput {
    id: String
    capabilities: [String!]
  }

  input UpdateWorkerInput {
    status: WorkerStatus
    current_task: String
    capabilities: [String!]
  }

  input AddDiscoveryInput {
    title: String!
    content: String!
    created_by: String
    tags: [String!]
  }

  input TaskFilter {
    status: TaskStatus
    priority: Int
    assigned_to: String
    tags: [String!]
  }

  type Query {
    # Tasks
    tasks(filter: TaskFilter): [Task!]!
    task(id: ID!): Task
    availableTasks: [Task!]!
    tasksByStatus(status: TaskStatus!): [Task!]!

    # Workers
    workers: [Worker!]!
    worker(id: ID!): Worker
    activeWorkers: [Worker!]!

    # Discoveries
    discoveries: [Discovery!]!
    discovery(id: ID!): Discovery
    searchDiscoveries(query: String!): [Discovery!]!

    # Status
    status: CoordinationStatus!
  }

  type Mutation {
    # Tasks
    createTask(input: CreateTaskInput!): Task!
    updateTask(id: ID!, input: UpdateTaskInput!): Task
    deleteTask(id: ID!): Boolean!
    claimTask(id: ID!, workerId: String!): Task
    startTask(id: ID!): Task
    completeTask(id: ID!, result: String): Task
    failTask(id: ID!, error: String): Task

    # Workers
    registerWorker(input: RegisterWorkerInput): Worker!
    updateWorker(id: ID!, input: UpdateWorkerInput!): Worker
    heartbeat(id: ID!): Worker
    unregisterWorker(id: ID!): Boolean!

    # Discoveries
    addDiscovery(input: AddDiscoveryInput!): Discovery!
    deleteDiscovery(id: ID!): Boolean!

    # Coordination
    initCoordination(goal: String!): Boolean!
  }

  type Subscription {
    taskCreated: Task!
    taskUpdated: Task!
    taskDeleted: ID!
    workerUpdated: Worker!
    discoveryAdded: Discovery!
  }
`;

// Data service
class DataService {
  private coordinationDir: string;

  constructor(coordinationDir: string) {
    this.coordinationDir = coordinationDir;
  }

  private ensureDir(dirPath: string): void {
    if (!fs.existsSync(dirPath)) {
      fs.mkdirSync(dirPath, { recursive: true });
    }
  }

  // Tasks
  getTasks(): any[] {
    const tasksPath = path.join(this.coordinationDir, "tasks.json");
    if (!fs.existsSync(tasksPath)) return [];
    try {
      return JSON.parse(fs.readFileSync(tasksPath, "utf-8")).tasks || [];
    } catch {
      return [];
    }
  }

  saveTasks(tasks: any[]): void {
    this.ensureDir(this.coordinationDir);
    const tasksPath = path.join(this.coordinationDir, "tasks.json");
    fs.writeFileSync(
      tasksPath,
      JSON.stringify({ tasks, updated_at: new Date().toISOString() }, null, 2),
    );
  }

  getTask(id: string): any {
    return this.getTasks().find((t) => t.id === id);
  }

  createTask(input: any): any {
    const tasks = this.getTasks();
    const task = {
      id: `task-${uuidv4()}`,
      description: input.description,
      status: "available",
      priority: input.priority || 3,
      dependencies: input.dependencies || [],
      tags: input.tags || [],
      created_at: new Date().toISOString(),
    };
    tasks.push(task);
    this.saveTasks(tasks);
    pubsub.publish(TASK_CREATED, { taskCreated: task });
    return task;
  }

  updateTask(id: string, input: any): any {
    const tasks = this.getTasks();
    const index = tasks.findIndex((t) => t.id === id);
    if (index === -1) return null;

    tasks[index] = {
      ...tasks[index],
      ...input,
      updated_at: new Date().toISOString(),
    };
    this.saveTasks(tasks);
    pubsub.publish(TASK_UPDATED, { taskUpdated: tasks[index] });
    return tasks[index];
  }

  deleteTask(id: string): boolean {
    const tasks = this.getTasks();
    const index = tasks.findIndex((t) => t.id === id);
    if (index === -1) return false;

    tasks.splice(index, 1);
    this.saveTasks(tasks);
    pubsub.publish(TASK_DELETED, { taskDeleted: id });
    return true;
  }

  // Workers
  getWorkers(): any[] {
    const workersPath = path.join(this.coordinationDir, "workers.json");
    if (!fs.existsSync(workersPath)) return [];
    try {
      return JSON.parse(fs.readFileSync(workersPath, "utf-8")).workers || [];
    } catch {
      return [];
    }
  }

  saveWorkers(workers: any[]): void {
    this.ensureDir(this.coordinationDir);
    const workersPath = path.join(this.coordinationDir, "workers.json");
    fs.writeFileSync(workersPath, JSON.stringify({ workers }, null, 2));
  }

  getWorker(id: string): any {
    return this.getWorkers().find((w) => w.id === id);
  }

  registerWorker(input: any): any {
    const workers = this.getWorkers();
    const worker = {
      id: input?.id || `worker-${uuidv4()}`,
      status: "idle",
      last_heartbeat: new Date().toISOString(),
      tasks_completed: 0,
      capabilities: input?.capabilities || [],
    };
    workers.push(worker);
    this.saveWorkers(workers);
    pubsub.publish(WORKER_UPDATED, { workerUpdated: worker });
    return worker;
  }

  updateWorker(id: string, input: any): any {
    const workers = this.getWorkers();
    const index = workers.findIndex((w) => w.id === id);
    if (index === -1) return null;

    workers[index] = { ...workers[index], ...input };
    this.saveWorkers(workers);
    pubsub.publish(WORKER_UPDATED, { workerUpdated: workers[index] });
    return workers[index];
  }

  // Discoveries
  getDiscoveries(): any[] {
    const discoveriesPath = path.join(this.coordinationDir, "discoveries.json");
    if (!fs.existsSync(discoveriesPath)) return [];
    try {
      return (
        JSON.parse(fs.readFileSync(discoveriesPath, "utf-8")).discoveries || []
      );
    } catch {
      return [];
    }
  }

  saveDiscoveries(discoveries: any[]): void {
    this.ensureDir(this.coordinationDir);
    const discoveriesPath = path.join(this.coordinationDir, "discoveries.json");
    fs.writeFileSync(discoveriesPath, JSON.stringify({ discoveries }, null, 2));
  }

  addDiscovery(input: any): any {
    const discoveries = this.getDiscoveries();
    const discovery = {
      id: `discovery-${uuidv4()}`,
      title: input.title,
      content: input.content,
      created_by: input.created_by || "graphql",
      created_at: new Date().toISOString(),
      tags: input.tags || [],
    };
    discoveries.push(discovery);
    this.saveDiscoveries(discoveries);
    pubsub.publish(DISCOVERY_ADDED, { discoveryAdded: discovery });
    return discovery;
  }

  deleteDiscovery(id: string): boolean {
    const discoveries = this.getDiscoveries();
    const index = discoveries.findIndex((d) => d.id === id);
    if (index === -1) return false;
    discoveries.splice(index, 1);
    this.saveDiscoveries(discoveries);
    return true;
  }

  // Status
  getStatus(): any {
    const tasks = this.getTasks();
    const workers = this.getWorkers();

    return {
      tasks: {
        total: tasks.length,
        available: tasks.filter((t) => t.status === "available").length,
        claimed: tasks.filter((t) => t.status === "claimed").length,
        in_progress: tasks.filter((t) => t.status === "in_progress").length,
        done: tasks.filter((t) => t.status === "done").length,
        failed: tasks.filter((t) => t.status === "failed").length,
      },
      workers: {
        total: workers.length,
        idle: workers.filter((w) => w.status === "idle").length,
        busy: workers.filter((w) => w.status === "busy").length,
        offline: workers.filter((w) => w.status === "offline").length,
      },
      timestamp: new Date().toISOString(),
    };
  }

  // Init coordination
  initCoordination(goal: string): boolean {
    this.ensureDir(this.coordinationDir);
    this.ensureDir(path.join(this.coordinationDir, "context"));
    this.ensureDir(path.join(this.coordinationDir, "results"));
    this.ensureDir(path.join(this.coordinationDir, "logs"));

    const masterPlanPath = path.join(this.coordinationDir, "master-plan.md");
    fs.writeFileSync(
      masterPlanPath,
      `# Master Plan\n\nGoal: ${goal}\n\nCreated: ${new Date().toISOString()}\n`,
    );

    this.saveTasks([]);
    this.saveWorkers([]);
    this.saveDiscoveries([]);

    return true;
  }
}

// Create resolvers
function createResolvers(service: DataService) {
  return {
    Query: {
      tasks: (_: any, { filter }: any) => {
        let tasks = service.getTasks();
        if (filter) {
          if (filter.status) {
            tasks = tasks.filter((t) => t.status === filter.status);
          }
          if (filter.priority) {
            tasks = tasks.filter((t) => t.priority === filter.priority);
          }
          if (filter.assigned_to) {
            tasks = tasks.filter((t) => t.assigned_to === filter.assigned_to);
          }
          if (filter.tags && filter.tags.length > 0) {
            tasks = tasks.filter(
              (t) =>
                t.tags &&
                filter.tags.some((tag: string) => t.tags.includes(tag)),
            );
          }
        }
        return tasks.sort((a, b) => a.priority - b.priority);
      },
      task: (_: any, { id }: any) => service.getTask(id),
      availableTasks: () =>
        service.getTasks().filter((t) => t.status === "available"),
      tasksByStatus: (_: any, { status }: any) =>
        service.getTasks().filter((t) => t.status === status),

      workers: () => service.getWorkers(),
      worker: (_: any, { id }: any) => service.getWorker(id),
      activeWorkers: () =>
        service.getWorkers().filter((w) => w.status !== "offline"),

      discoveries: () => service.getDiscoveries(),
      discovery: (_: any, { id }: any) =>
        service.getDiscoveries().find((d) => d.id === id),
      searchDiscoveries: (_: any, { query }: any) =>
        service
          .getDiscoveries()
          .filter(
            (d) =>
              d.title.toLowerCase().includes(query.toLowerCase()) ||
              d.content.toLowerCase().includes(query.toLowerCase()),
          ),

      status: () => service.getStatus(),
    },

    Mutation: {
      createTask: (_: any, { input }: any) => service.createTask(input),
      updateTask: (_: any, { id, input }: any) => service.updateTask(id, input),
      deleteTask: (_: any, { id }: any) => service.deleteTask(id),
      claimTask: (_: any, { id, workerId }: any) => {
        const task = service.getTask(id);
        if (!task || task.status !== "available") return null;
        return service.updateTask(id, {
          status: "claimed",
          assigned_to: workerId,
        });
      },
      startTask: (_: any, { id }: any) =>
        service.updateTask(id, { status: "in_progress" }),
      completeTask: (_: any, { id, result }: any) =>
        service.updateTask(id, { status: "done", result }),
      failTask: (_: any, { id, error }: any) =>
        service.updateTask(id, { status: "failed", error }),

      registerWorker: (_: any, { input }: any) => service.registerWorker(input),
      updateWorker: (_: any, { id, input }: any) =>
        service.updateWorker(id, input),
      heartbeat: (_: any, { id }: any) =>
        service.updateWorker(id, {
          last_heartbeat: new Date().toISOString(),
        }),
      unregisterWorker: (_: any, { id }: any) => {
        const workers = service.getWorkers();
        const index = workers.findIndex((w) => w.id === id);
        if (index === -1) return false;
        workers.splice(index, 1);
        service.saveWorkers(workers);
        return true;
      },

      addDiscovery: (_: any, { input }: any) => service.addDiscovery(input),
      deleteDiscovery: (_: any, { id }: any) => service.deleteDiscovery(id),

      initCoordination: (_: any, { goal }: any) =>
        service.initCoordination(goal),
    },

    Subscription: {
      taskCreated: {
        subscribe: () => pubsub.asyncIterator([TASK_CREATED]),
      },
      taskUpdated: {
        subscribe: () => pubsub.asyncIterator([TASK_UPDATED]),
      },
      taskDeleted: {
        subscribe: () => pubsub.asyncIterator([TASK_DELETED]),
      },
      workerUpdated: {
        subscribe: () => pubsub.asyncIterator([WORKER_UPDATED]),
      },
      discoveryAdded: {
        subscribe: () => pubsub.asyncIterator([DISCOVERY_ADDED]),
      },
    },
  };
}

// Create and start server
export async function createGraphQLServer(
  coordinationDir: string,
  port: number = 3003,
) {
  const service = new DataService(coordinationDir);
  const resolvers = createResolvers(service);

  const schema = makeExecutableSchema({ typeDefs, resolvers });

  const app = express();
  const httpServer = createServer(app);

  // WebSocket server for subscriptions
  const wsServer = new WebSocketServer({
    server: httpServer,
    path: "/graphql",
  });

  const serverCleanup = useServer({ schema }, wsServer);

  const server = new ApolloServer({
    schema,
    plugins: [
      ApolloServerPluginDrainHttpServer({ httpServer }),
      {
        async serverWillStart() {
          return {
            async drainServer() {
              await serverCleanup.dispose();
            },
          };
        },
      },
    ],
  });

  await server.start();

  app.use(
    "/graphql",
    cors<cors.CorsRequest>(),
    express.json(),
    expressMiddleware(server),
  );

  // Health check
  app.get("/health", (req, res) => {
    res.json({ status: "ok", timestamp: new Date().toISOString() });
  });

  return new Promise<void>((resolve) => {
    httpServer.listen(port, () => {
      console.log(`GraphQL server running at http://localhost:${port}/graphql`);
      console.log(`WebSocket subscriptions at ws://localhost:${port}/graphql`);
      resolve();
    });
  });
}

// Main entry point
if (require.main === module) {
  const coordinationDir = process.env.COORDINATION_DIR || ".coordination";
  const port = parseInt(process.env.GRAPHQL_PORT || "3003");

  createGraphQLServer(coordinationDir, port).catch(console.error);
}

export { DataService, typeDefs, createResolvers };
