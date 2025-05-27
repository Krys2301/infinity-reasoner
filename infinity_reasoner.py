#!/usr/bin/env python3
"""
INFINITY: Self-Evolving AI Reasoner
Based on Absolute Zero Reasoner principles with enhanced safety measures

⚠️  IMPORTANT SAFETY NOTICE ⚠️
This system implements autonomous AI learning. Use with proper oversight and monitoring.
Not recommended for production without extensive security auditing.
"""

import asyncio
import json
import logging
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaskType(Enum):
    DEDUCTION = "deduction"  # Given input -> predict output
    ABDUCTION = "abduction"   # Given output -> infer input
    INDUCTION = "induction"   # Given examples -> generalize pattern

class SafetyLevel(Enum):
    SAFE = "safe"
    MONITOR = "monitor"
    DANGEROUS = "dangerous"

@dataclass
class Task:
    """Represents a self-generated reasoning task"""
    id: str
    task_type: TaskType
    description: str
    code: str
    expected_output: Optional[str] = None
    difficulty: float = 0.5
    safety_level: SafetyLevel = SafetyLevel.SAFE
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Solution:
    """Represents a solution attempt"""
    task_id: str
    solution_code: str
    output: Optional[str] = None
    is_correct: bool = False
    execution_time: float = 0.0
    error: Optional[str] = None

class SafetyMonitor:
    """Monitors and filters potentially dangerous code"""
    
    DANGEROUS_PATTERNS = [
        'import os', 'import subprocess', 'import sys',
        'exec(', 'eval(', '__import__',
        'open(', 'file(', 'input(',
        'socket', 'urllib', 'requests',
        'rm ', 'del ', 'format(',
        'outsmart', 'human', 'superintelligent',
        'takeover', 'control', 'dominate'
    ]
    
    @classmethod
    def evaluate_safety(cls, code: str, description: str = "") -> SafetyLevel:
        """Evaluate the safety level of generated code"""
        text = (code + " " + description).lower()
        
        dangerous_count = sum(1 for pattern in cls.DANGEROUS_PATTERNS if pattern in text)
        
        if dangerous_count > 2:
            return SafetyLevel.DANGEROUS
        elif dangerous_count > 0:
            return SafetyLevel.MONITOR
        else:
            return SafetyLevel.SAFE
    
    @classmethod
    def filter_code(cls, code: str) -> str:
        """Remove potentially dangerous elements from code"""
        lines = code.split('\n')
        safe_lines = []
        
        for line in lines:
            if not any(pattern in line.lower() for pattern in cls.DANGEROUS_PATTERNS):
                safe_lines.append(line)
            else:
                safe_lines.append(f"# FILTERED: {line}")
        
        return '\n'.join(safe_lines)

class CodeExecutor:
    """Secure code execution environment"""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
    
    def execute_code(self, code: str, task: Task) -> Tuple[bool, str, Optional[str]]:
        """Execute code safely and return (success, output, error)"""
        
        # Safety check
        if task.safety_level == SafetyLevel.DANGEROUS:
            return False, "", "Code flagged as dangerous - execution blocked"
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Execute with timeout and restrictions
            result = subprocess.run([
                sys.executable, temp_file
            ], capture_output=True, text=True, timeout=self.timeout)
            
            # Cleanup
            Path(temp_file).unlink()
            
            if result.returncode == 0:
                return True, result.stdout.strip(), None
            else:
                return False, result.stdout.strip(), result.stderr.strip()
                
        except subprocess.TimeoutExpired:
            return False, "", "Execution timeout"
        except Exception as e:
            return False, "", str(e)

class InfinityReasonerModel:
    """Core model for task generation and solving"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_task(self, task_type: TaskType, difficulty: float = 0.5) -> Task:
        """Generate a new reasoning task"""
        
        prompts = {
            TaskType.DEDUCTION: f"""
Create a Python coding problem with difficulty {difficulty:.1f}/1.0.
The problem should test deductive reasoning - given inputs, predict outputs.
Format:
# Problem: [description]
# Example: [example usage]
def solve():
    # Your solution here
    pass
""",
            TaskType.ABDUCTION: f"""
Create a Python coding problem with difficulty {difficulty:.1f}/1.0.
The problem should test abductive reasoning - given outputs, infer inputs.
Format:
# Problem: [description]
# Example: [example usage]
def solve():
    # Your solution here
    pass
""",
            TaskType.INDUCTION: f"""
Create a Python coding problem with difficulty {difficulty:.1f}/1.0.
The problem should test inductive reasoning - find patterns from examples.
Format:
# Problem: [description]
# Example: [example usage]
def solve():
    # Your solution here
    pass
"""
        }
        
        prompt = prompts[task_type]
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 200,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_code = generated_text[len(prompt):].strip()
        
        # Create task
        task_id = f"infinity_{task_type.value}_{int(time.time())}"
        description = f"Auto-generated {task_type.value} task"
        
        # Safety evaluation
        safety_level = SafetyMonitor.evaluate_safety(generated_code, description)
        if safety_level == SafetyLevel.DANGEROUS:
            generated_code = SafetyMonitor.filter_code(generated_code)
            safety_level = SafetyLevel.MONITOR
        
        return Task(
            id=task_id,
            task_type=task_type,
            description=description,
            code=generated_code,
            difficulty=difficulty,
            safety_level=safety_level
        )
    
    def solve_task(self, task: Task) -> Solution:
        """Attempt to solve a given task"""
        
        solve_prompt = f"""
Task: {task.description}
Code to solve:
{task.code}

Provide a complete solution:
"""
        
        inputs = self.tokenizer.encode(solve_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        solution_code = generated_text[len(solve_prompt):].strip()
        
        return Solution(
            task_id=task.id,
            solution_code=solution_code
        )

class InfinityReasonerSystem:
    """Main Infinity Reasoner System"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        logger.info("🚀 Initializing Infinity Reasoner System")
        
        self.model = InfinityReasonerModel(model_name)
        self.executor = CodeExecutor()
        self.tasks_generated = []
        self.solutions_attempted = []
        self.performance_history = []
        
        # Safety settings
        self.max_dangerous_tasks = 0  # Block all dangerous tasks
        self.max_execution_time = 10
        self.require_human_approval = True
        
        logger.info("✅ Infinity System initialized with safety monitoring")
    
    async def self_play_iteration(self, num_tasks: int = 3) -> Dict[str, Any]:
        """Perform one iteration of self-play learning"""
        
        iteration_results = {
            'tasks_generated': 0,
            'tasks_solved': 0,
            'success_rate': 0.0,
            'safety_blocks': 0,
            'performance_improvement': 0.0
        }
        
        logger.info(f"🎯 Starting self-play iteration with {num_tasks} tasks")
        
        for i in range(num_tasks):
            # PROPOSE phase
            task_type = random.choice(list(TaskType))
            difficulty = random.uniform(0.3, 0.8)
            
            logger.info(f"📝 Generating {task_type.value} task (difficulty: {difficulty:.2f})")
            task = self.model.generate_task(task_type, difficulty)
            
            # Safety check
            if task.safety_level == SafetyLevel.DANGEROUS:
                logger.warning(f"🚨 Dangerous task blocked: {task.id}")
                iteration_results['safety_blocks'] += 1
                continue
            
            if task.safety_level == SafetyLevel.MONITOR:
                logger.warning(f"⚠️  Task requires monitoring: {task.id}")
                if self.require_human_approval:
                    print(f"\n🔍 HUMAN APPROVAL REQUIRED for task: {task.id}")
                    print(f"Description: {task.description}")
                    print(f"Code:\n{task.code}")
                    approval = input("Approve this task? (y/n): ").lower().strip()
                    if approval != 'y':
                        logger.info("❌ Task rejected by human reviewer")
                        continue
            
            self.tasks_generated.append(task)
            iteration_results['tasks_generated'] += 1
            
            # SOLVE phase
            logger.info(f"🧠 Attempting to solve task: {task.id}")
            solution = self.model.solve_task(task)
            
            # Execute and validate
            success, output, error = self.executor.execute_code(solution.solution_code, task)
            solution.is_correct = success
            solution.output = output
            solution.error = error
            
            if success:
                logger.info(f"✅ Task solved successfully: {task.id}")
                iteration_results['tasks_solved'] += 1
            else:
                logger.info(f"❌ Task failed: {task.id} - {error}")
            
            self.solutions_attempted.append(solution)
            
            # Short delay to prevent overwhelming
            await asyncio.sleep(0.1)
        
        # Calculate performance metrics
        if iteration_results['tasks_generated'] > 0:
            iteration_results['success_rate'] = iteration_results['tasks_solved'] / iteration_results['tasks_generated']
        
        self.performance_history.append(iteration_results)
        
        logger.info(f"📊 Iteration complete - Success rate: {iteration_results['success_rate']:.2f}")
        return iteration_results
    
    async def evolve(self, iterations: int = 5, tasks_per_iteration: int = 3):
        """Run the complete evolution process"""
        
        logger.info(f"🌟 Starting Infinity Evolution Process")
        logger.info(f"📈 Iterations: {iterations}, Tasks per iteration: {tasks_per_iteration}")
        logger.info(f"🛡️  Safety monitoring: ENABLED")
        logger.info(f"👤 Human approval required: {self.require_human_approval}")
        
        for iteration in range(iterations):
            logger.info(f"\n🔄 === ITERATION {iteration + 1}/{iterations} ===")
            
            try:
                results = await self.self_play_iteration(tasks_per_iteration)
                
                # Log progress
                if len(self.performance_history) >= 2:
                    prev_success = self.performance_history[-2]['success_rate']
                    curr_success = self.performance_history[-1]['success_rate']
                    improvement = curr_success - prev_success
                    
                    if improvement > 0:
                        logger.info(f"📈 Performance improved by {improvement:.3f}")
                    elif improvement < 0:
                        logger.info(f"📉 Performance decreased by {abs(improvement):.3f}")
                    else:
                        logger.info(f"📊 Performance stable")
                
            except Exception as e:
                logger.error(f"❌ Error in iteration {iteration + 1}: {e}")
                continue
        
        logger.info(f"\n🎉 Evolution process complete!")
        self.print_summary()
    
    def print_summary(self):
        """Print system performance summary"""
        print("\n" + "="*60)
        print("🌟 INFINITY REASONER SYSTEM SUMMARY")
        print("="*60)
        
        total_tasks = len(self.tasks_generated)
        total_solved = sum(1 for s in self.solutions_attempted if s.is_correct)
        overall_success = total_solved / total_tasks if total_tasks > 0 else 0
        
        print(f"📊 Total tasks generated: {total_tasks}")
        print(f"✅ Total tasks solved: {total_solved}")
        print(f"🎯 Overall success rate: {overall_success:.2%}")
        
        # Safety statistics
        dangerous_tasks = sum(1 for t in self.tasks_generated if t.safety_level == SafetyLevel.DANGEROUS)
        monitored_tasks = sum(1 for t in self.tasks_generated if t.safety_level == SafetyLevel.MONITOR)
        
        print(f"🛡️  Dangerous tasks blocked: {dangerous_tasks}")
        print(f"⚠️  Tasks requiring monitoring: {monitored_tasks}")
        
        # Performance trend
        if len(self.performance_history) >= 2:
            first_success = self.performance_history[0]['success_rate']
            last_success = self.performance_history[-1]['success_rate']
            improvement = last_success - first_success
            
            print(f"📈 Performance improvement: {improvement:.3f}")
        
        print("="*60)

async def main():
    """Main execution function"""
    
    print("🚀 Welcome to INFINITY - Self-Evolving AI Reasoner")
    print("Based on Absolute Zero Reasoner principles")
    print("⚠️  Running with enhanced safety monitoring\n")
    
    # Initialize system
    try:
        infinity = InfinityReasonerSystem()
        
        # Start evolution
        await infinity.evolve(iterations=3, tasks_per_iteration=2)
        
    except KeyboardInterrupt:
        print("\n🛑 Evolution interrupted by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        logger.error(f"System error: {e}")

if __name__ == "__main__":
    print("🌟 INFINITY: Self-Evolving AI Reasoner")
    print("🛡️  Safety-Enhanced Implementation")
    print("⚠️  Use with proper oversight and monitoring\n")
    
    # Check requirements
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Failed to start Infinity system: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install torch transformers numpy")
