#!/usr/bin/env node

import { Command } from "commander";
import chalk from "chalk";
import ora from "ora";
import inquirer from "inquirer";
import * as fs from "fs";
import * as path from "path";
import { PluginManager } from "./index";
import { config } from "dotenv";

config();

const program = new Command();
const coordinationDir = process.env.COORDINATION_DIR || ".coordination";
const pluginManager = new PluginManager(coordinationDir);

program
  .name("coord")
  .description("Claude Multi-Agent Coordinator CLI with plugin support")
  .version("1.0.0");

// Plugin management commands
const pluginCmd = program.command("plugin").description("Manage plugins");

pluginCmd
  .command("list")
  .description("List installed plugins")
  .action(async () => {
    const spinner = ora("Loading plugins...").start();
    await pluginManager.loadAllPlugins();
    spinner.stop();

    const plugins = pluginManager.getPlugins();

    if (plugins.length === 0) {
      console.log(chalk.yellow("No plugins installed."));
      return;
    }

    console.log(chalk.bold("\nInstalled Plugins:\n"));

    for (const plugin of plugins) {
      console.log(
        chalk.cyan(`  ${plugin.metadata.name}`) +
          chalk.gray(` v${plugin.metadata.version}`),
      );
      console.log(chalk.gray(`    ${plugin.metadata.description}`));

      if (plugin.commands) {
        const cmdCount = Object.keys(plugin.commands).length;
        console.log(chalk.gray(`    Commands: ${cmdCount}`));
      }

      if (plugin.hooks) {
        const hookCount = Object.keys(plugin.hooks).length;
        console.log(chalk.gray(`    Hooks: ${hookCount}`));
      }

      console.log();
    }
  });

pluginCmd
  .command("install <source>")
  .description("Install a plugin from path or npm")
  .action(async (source: string) => {
    const spinner = ora(`Installing plugin from ${source}...`).start();

    try {
      const success = await pluginManager.installPlugin(source);
      if (success) {
        spinner.succeed(chalk.green("Plugin installed successfully!"));
      } else {
        spinner.fail(chalk.red("Failed to install plugin."));
      }
    } catch (error) {
      spinner.fail(chalk.red(`Error: ${error}`));
    }
  });

pluginCmd
  .command("uninstall <name>")
  .description("Uninstall a plugin")
  .action(async (name: string) => {
    const { confirm } = await inquirer.prompt([
      {
        type: "confirm",
        name: "confirm",
        message: `Are you sure you want to uninstall ${name}?`,
        default: false,
      },
    ]);

    if (!confirm) {
      console.log(chalk.yellow("Cancelled."));
      return;
    }

    const spinner = ora(`Uninstalling ${name}...`).start();

    try {
      const success = await pluginManager.uninstallPlugin(name);
      if (success) {
        spinner.succeed(chalk.green("Plugin uninstalled successfully!"));
      } else {
        spinner.fail(chalk.red("Plugin not found."));
      }
    } catch (error) {
      spinner.fail(chalk.red(`Error: ${error}`));
    }
  });

pluginCmd
  .command("info <name>")
  .description("Show plugin information")
  .action(async (name: string) => {
    await pluginManager.loadAllPlugins();
    const plugin = pluginManager.getPlugin(name);

    if (!plugin) {
      console.log(chalk.red(`Plugin not found: ${name}`));
      return;
    }

    console.log(chalk.bold(`\n${plugin.metadata.name}\n`));
    console.log(chalk.gray(`Version: ${plugin.metadata.version}`));
    console.log(chalk.gray(`Description: ${plugin.metadata.description}`));

    if (plugin.metadata.author) {
      console.log(chalk.gray(`Author: ${plugin.metadata.author}`));
    }

    if (plugin.metadata.license) {
      console.log(chalk.gray(`License: ${plugin.metadata.license}`));
    }

    if (plugin.metadata.homepage) {
      console.log(chalk.gray(`Homepage: ${plugin.metadata.homepage}`));
    }

    if (plugin.commands) {
      console.log(chalk.bold("\nCommands:"));
      for (const [cmdName, cmd] of Object.entries(plugin.commands)) {
        console.log(chalk.cyan(`  ${name}:${cmdName}`));
        console.log(chalk.gray(`    ${cmd.description}`));
      }
    }

    if (plugin.hooks) {
      console.log(chalk.bold("\nHooks:"));
      for (const hookName of Object.keys(plugin.hooks)) {
        console.log(chalk.cyan(`  ${hookName}`));
      }
    }

    console.log();
  });

pluginCmd
  .command("commands")
  .description("List all available plugin commands")
  .action(async () => {
    await pluginManager.loadAllPlugins();
    const commands = pluginManager.getAllCommands();

    if (commands.size === 0) {
      console.log(chalk.yellow("No plugin commands available."));
      return;
    }

    console.log(chalk.bold("\nAvailable Plugin Commands:\n"));

    for (const [cmdPath, { plugin, command }] of commands) {
      console.log(chalk.cyan(`  ${cmdPath}`));
      console.log(chalk.gray(`    ${command.description}`));

      if (command.options) {
        for (const opt of command.options) {
          console.log(chalk.gray(`    ${opt.flags} - ${opt.description}`));
        }
      }

      console.log();
    }
  });

// Execute plugin command
pluginCmd
  .command("run <command>")
  .description("Run a plugin command (format: plugin:command)")
  .option("-a, --args <args>", "JSON arguments for the command")
  .action(async (command: string, options) => {
    await pluginManager.loadAllPlugins();

    try {
      let args = {};
      if (options.args) {
        args = JSON.parse(options.args);
      }

      await pluginManager.executeCommand(command, args);
    } catch (error) {
      console.error(chalk.red(`Error: ${error}`));
    }
  });

// Create plugin scaffold
pluginCmd
  .command("create <name>")
  .description("Create a new plugin from template")
  .option("-d, --dir <directory>", "Output directory", ".")
  .action(async (name: string, options) => {
    const pluginDir = path.join(options.dir, name);

    if (fs.existsSync(pluginDir)) {
      console.log(chalk.red(`Directory already exists: ${pluginDir}`));
      return;
    }

    const spinner = ora("Creating plugin scaffold...").start();

    try {
      // Create directory structure
      fs.mkdirSync(pluginDir, { recursive: true });
      fs.mkdirSync(path.join(pluginDir, "src"), { recursive: true });

      // Create package.json
      const packageJson = {
        name: name,
        version: "1.0.0",
        description: `${name} plugin for Claude Coordinator`,
        main: "index.js",
        scripts: {
          build: "tsc",
          dev: "ts-node src/index.ts",
        },
        engines: {
          coordinator: "^1.0.0",
        },
        keywords: ["claude-coordinator", "plugin"],
        devDependencies: {
          typescript: "^5.3.0",
          "ts-node": "^10.9.2",
        },
      };

      fs.writeFileSync(
        path.join(pluginDir, "package.json"),
        JSON.stringify(packageJson, null, 2),
      );

      // Create tsconfig.json
      const tsconfig = {
        compilerOptions: {
          target: "ES2022",
          module: "commonjs",
          outDir: "./",
          rootDir: "./src",
          strict: true,
          esModuleInterop: true,
        },
        include: ["src/**/*"],
      };

      fs.writeFileSync(
        path.join(pluginDir, "tsconfig.json"),
        JSON.stringify(tsconfig, null, 2),
      );

      // Create index.ts template
      const indexTs = `// ${name} plugin for Claude Coordinator

import { PluginHooks, PluginCommands, PluginContext } from '@claude-coordinator/cli-plugins';

export const hooks: PluginHooks = {
  onLoad: async (context: PluginContext) => {
    context.logger.info('${name} plugin loaded!');
  },

  onUnload: async (context: PluginContext) => {
    context.logger.info('${name} plugin unloaded!');
  },

  onTaskCreated: async (task: any, context: PluginContext) => {
    context.logger.debug(\`Task created: \${task.id}\`);
  },

  onTaskCompleted: async (task: any, context: PluginContext) => {
    context.logger.debug(\`Task completed: \${task.id}\`);
  },
};

export const commands: PluginCommands = {
  hello: {
    description: 'Say hello from ${name} plugin',
    options: [
      {
        flags: '-n, --name <name>',
        description: 'Name to greet',
        default: 'World',
      },
    ],
    action: async (args: any, context: PluginContext) => {
      const name = args.name || 'World';
      context.logger.info(\`Hello, \${name}!\`);
    },
  },

  'task-count': {
    description: 'Count tasks by status',
    action: async (args: any, context: PluginContext) => {
      const tasks = await context.api.getTasks();
      const counts: Record<string, number> = {};

      for (const task of tasks) {
        counts[task.status] = (counts[task.status] || 0) + 1;
      }

      context.logger.info('Task counts by status:');
      for (const [status, count] of Object.entries(counts)) {
        context.logger.info(\`  \${status}: \${count}\`);
      }
    },
  },
};
`;

      fs.writeFileSync(path.join(pluginDir, "src", "index.ts"), indexTs);

      // Create README
      const readme = `# ${name}

A plugin for Claude Multi-Agent Coordinator.

## Installation

\`\`\`bash
coord plugin install ./${name}
\`\`\`

## Commands

- \`${name}:hello\` - Say hello
- \`${name}:task-count\` - Count tasks by status

## Hooks

- \`onLoad\` - Called when plugin is loaded
- \`onUnload\` - Called when plugin is unloaded
- \`onTaskCreated\` - Called when a task is created
- \`onTaskCompleted\` - Called when a task is completed

## Development

\`\`\`bash
npm install
npm run build
\`\`\`
`;

      fs.writeFileSync(path.join(pluginDir, "README.md"), readme);

      spinner.succeed(chalk.green(`Plugin scaffold created at ${pluginDir}`));
      console.log(chalk.gray("\nNext steps:"));
      console.log(chalk.gray(`  cd ${pluginDir}`));
      console.log(chalk.gray("  npm install"));
      console.log(chalk.gray("  npm run build"));
      console.log(chalk.gray(`  coord plugin install ${pluginDir}`));
    } catch (error) {
      spinner.fail(chalk.red(`Error: ${error}`));
    }
  });

// Task commands
const taskCmd = program.command("task").description("Task operations");

taskCmd
  .command("list")
  .description("List all tasks")
  .option("-s, --status <status>", "Filter by status")
  .action(async (options) => {
    await pluginManager.loadAllPlugins();

    const tasksPath = path.join(coordinationDir, "tasks.json");
    if (!fs.existsSync(tasksPath)) {
      console.log(chalk.yellow("No tasks found."));
      return;
    }

    const data = JSON.parse(fs.readFileSync(tasksPath, "utf-8"));
    let tasks = data.tasks || [];

    if (options.status) {
      tasks = tasks.filter((t: any) => t.status === options.status);
    }

    if (tasks.length === 0) {
      console.log(chalk.yellow("No tasks found."));
      return;
    }

    console.log(chalk.bold("\nTasks:\n"));

    for (const task of tasks) {
      const statusColor =
        task.status === "done"
          ? chalk.green
          : task.status === "failed"
            ? chalk.red
            : task.status === "in_progress"
              ? chalk.blue
              : chalk.yellow;

      console.log(
        `  ${chalk.gray(task.id)} ${statusColor(`[${task.status}]`)} P${task.priority}`,
      );
      console.log(`    ${task.description}`);
    }

    console.log();
  });

taskCmd
  .command("create <description>")
  .description("Create a new task")
  .option("-p, --priority <priority>", "Task priority (1-5)", "3")
  .action(async (description: string, options) => {
    await pluginManager.loadAllPlugins();

    const tasksPath = path.join(coordinationDir, "tasks.json");
    let data = { tasks: [] as any[] };

    if (fs.existsSync(tasksPath)) {
      data = JSON.parse(fs.readFileSync(tasksPath, "utf-8"));
    }

    const task = {
      id: `task-${Date.now()}`,
      description,
      status: "available",
      priority: parseInt(options.priority),
      created_at: new Date().toISOString(),
    };

    data.tasks.push(task);

    if (!fs.existsSync(coordinationDir)) {
      fs.mkdirSync(coordinationDir, { recursive: true });
    }

    fs.writeFileSync(tasksPath, JSON.stringify(data, null, 2));

    // Execute hooks
    await pluginManager.executeHook("onTaskCreated", task);

    console.log(chalk.green(`Task created: ${task.id}`));
  });

// Status command
program
  .command("status")
  .description("Show coordination status")
  .action(async () => {
    await pluginManager.loadAllPlugins();

    const tasksPath = path.join(coordinationDir, "tasks.json");
    const workersPath = path.join(coordinationDir, "workers.json");

    const tasks = fs.existsSync(tasksPath)
      ? JSON.parse(fs.readFileSync(tasksPath, "utf-8")).tasks || []
      : [];

    const workers = fs.existsSync(workersPath)
      ? JSON.parse(fs.readFileSync(workersPath, "utf-8")).workers || []
      : [];

    console.log(chalk.bold("\nCoordination Status\n"));

    console.log(chalk.cyan("Tasks:"));
    console.log(
      `  Available: ${tasks.filter((t: any) => t.status === "available").length}`,
    );
    console.log(
      `  Claimed: ${tasks.filter((t: any) => t.status === "claimed").length}`,
    );
    console.log(
      `  In Progress: ${tasks.filter((t: any) => t.status === "in_progress").length}`,
    );
    console.log(
      `  Done: ${tasks.filter((t: any) => t.status === "done").length}`,
    );
    console.log(
      `  Failed: ${tasks.filter((t: any) => t.status === "failed").length}`,
    );
    console.log(`  Total: ${tasks.length}`);

    console.log(chalk.cyan("\nWorkers:"));
    console.log(
      `  Idle: ${workers.filter((w: any) => w.status === "idle").length}`,
    );
    console.log(
      `  Busy: ${workers.filter((w: any) => w.status === "busy").length}`,
    );
    console.log(
      `  Offline: ${workers.filter((w: any) => w.status === "offline").length}`,
    );
    console.log(`  Total: ${workers.length}`);

    const plugins = pluginManager.getPlugins();
    console.log(chalk.cyan("\nPlugins:"));
    console.log(`  Loaded: ${plugins.length}`);

    console.log();
  });

// Parse and execute
async function main() {
  await pluginManager.loadAllPlugins();

  // Add plugin commands to program
  const commands = pluginManager.getAllCommands();
  for (const [cmdPath, { command }] of commands) {
    const cmd = program.command(cmdPath).description(command.description);

    if (command.options) {
      for (const opt of command.options) {
        cmd.option(opt.flags, opt.description, opt.default);
      }
    }

    cmd.action(async (options: any) => {
      await pluginManager.executeCommand(cmdPath, options);
    });
  }

  program.parse();
}

main().catch(console.error);
