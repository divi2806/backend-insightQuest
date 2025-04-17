from fastapi import FastAPI, HTTPException, Request, status, APIRouter, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from datetime import datetime
from web3 import Web3
from eth_account import Account
import json
from typing import Optional, Dict, Any, List
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai
from decouple import config
import re

# Initialize FastAPI app
app = FastAPI(title="LeetCode and Quiz Verification Service")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Firebase Admin SDK
try:
    service_account_path = "serviceAccountKey.json"
    cred = credentials.Certificate(service_account_path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase initialized successfully")
except Exception as error:
    print(f"Firebase initialization error: {error}")
    # Create a fallback for Firebase if initialization fails
    db = None

# Initialize Gemini API
GEMINI_API_KEY = config("GEMINI_API_KEY", default="AIzaSyC9YKF89cnfSAAzM6TilPY29Ea9LeiIf8s")
genai.configure(api_key=GEMINI_API_KEY)

# LeetCode GraphQL endpoint
LEETCODE_API = "https://leetcode.com/graphql"

# GraphQL query to get recent accepted submissions
RECENT_AC_SUBMISSIONS_QUERY = """
query recentAcSubmissions($username: String!, $limit: Int!) {
  recentAcSubmissionList(username: $username, limit: $limit) {
    id
    title
    titleSlug
    timestamp
  }
}
"""

# GraphQL query to get problem difficulty
PROBLEM_DIFFICULTY_QUERY = """
query problemData($titleSlug: String!) {
  question(titleSlug: $titleSlug) {
    difficulty
    title
  }
}
"""

# GraphQL query to fetch a user's LeetCode profile
USER_PROFILE_QUERY = """
query userProfile($username: String!) {
  matchedUser(username: $username) {
    username
    profile {
      realName
      aboutMe
      userAvatar
      ranking
    }
  }
}
"""

# Ethereum configuration
WEB3_PROVIDER = "https://sepolia.infura.io/v3/a975efc7b9154c9882437558faef96b7"
TOKEN_CONTRACT_ADDRESS = "0x4f87b9dE9Dd8F13EC323C0eDfb082c1363BafBb7"
TREASURY_PRIVATE_KEY = "ea5f30606c38ec9abf62cd19a3cb84db5edb1c5cd1bf9406c9a0f7cd1a26501a"

# ERC-20 ABI for token transfers
ERC20_ABI = [
    {
        "constant": False,
        "inputs": [
            {"name": "_to", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "transfer",
        "outputs": [{"name": "success", "type": "bool"}],
        "payable": False,
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    }
]

# Initialize Web3
web3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER))
token_contract = web3.eth.contract(address=web3.to_checksum_address(TOKEN_CONTRACT_ADDRESS), abi=ERC20_ABI)
treasury_account = Account.from_key(TREASURY_PRIVATE_KEY)

#############################################
# Request Models
#############################################

class VerifyLeetCodeTaskRequest(BaseModel):
    leetcodeUsername: str
    problemUrl: str
    walletAddress: str
    taskCreatedAt: Optional[str] = None

class LeetCodeVerificationRequest(BaseModel):
    username: str

class AlternativeVerificationRequest(BaseModel):
    username: str
    walletAddress: str

class QuizRequest(BaseModel):
    topic: str
    task_title: str
    task_description: Optional[str] = None
    num_questions: int = 5

class QuizResponse(BaseModel):
    success: bool
    questions: List[Dict[str, Any]] = []
    message: Optional[str] = None

class QuizAnswerRequest(BaseModel):
    userId: str
    taskId: str
    answers: List[Dict[str, Any]]
    walletAddress: str

class QuizVerificationResponse(BaseModel):
    success: bool
    score: int
    totalQuestions: int
    passed: bool
    reward: Optional[int] = None
    txHash: Optional[str] = None
    message: Optional[str] = None

#############################################
# Helper Functions
#############################################

# Extract problem slug from LeetCode URL
def extract_problem_slug(url: str) -> Optional[str]:
    try:
        if "leetcode.com/problems/" not in url:
            return None
        
        # Extract the slug from URL like https://leetcode.com/problems/two-sum/
        parts = url.split("/problems/")
        if len(parts) < 2:
            return None
        
        slug = parts[1].split("/")[0]
        return slug
    except Exception as e:
        print(f"Error extracting problem slug: {e}")
        return None

# Calculate reward based on problem difficulty
def calculate_reward(difficulty: str) -> int:
    difficulty = difficulty.lower()
    if difficulty == "hard":
        return 20
    elif difficulty == "medium":
        return 10
    else:  # Easy
        return 5

# Transfer tokens to user
async def transfer_tokens(recipient_address: str, amount: int) -> Dict[str, Any]:
    try:
        # Get token decimals
        decimals = token_contract.functions.decimals().call()
        
        # Convert amount to wei (with proper decimals)
        amount_in_wei = amount * (10 ** decimals)
        
        # Build the transaction
        nonce = web3.eth.get_transaction_count(treasury_account.address)
        
        tx = token_contract.functions.transfer(
            web3.to_checksum_address(recipient_address),
            amount_in_wei
        ).build_transaction({
            'chainId': 11155111,  # Sepolia chain ID
            'gas': 100000,
            'gasPrice': web3.to_wei('50', 'gwei'),
            'nonce': nonce,
        })
        
        # Sign and send the transaction
        signed_tx = web3.eth.account.sign_transaction(tx, treasury_account.key)
        # Fix: Access the correct attribute based on Web3.py version
        raw_tx_data = signed_tx.rawTransaction if hasattr(signed_tx, 'rawTransaction') else signed_tx.raw_transaction
        tx_hash = web3.eth.send_raw_transaction(raw_tx_data)
        
        # Wait for transaction receipt
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        
        return {
            "success": receipt.status == 1,
            "txHash": web3.to_hex(tx_hash),
            "blockNumber": receipt.blockNumber
        }
    except Exception as e:
        print(f"Error transferring tokens: {e}")
        return {"success": False, "error": str(e)}

#############################################
# Quiz Endpoints
#############################################

@app.post("/api/generate-quiz", response_model=QuizResponse)
async def generate_quiz(request: QuizRequest):
    try:
        # Create the prompt for Gemini - using structured formatting instructions
        prompt = f"""
        Generate {request.num_questions} quiz questions about the topic: {request.topic}
        
        The quiz is for a task titled: "{request.task_title}"
        
        Additional context: {request.task_description or ""}
        
        Make sure the questions test understanding of key concepts related to this topic.
        Each question should have 1 correct answer and 3 incorrect answers.
        Vary the difficulty level of questions.
        
        Format your response as a JSON object with the following structure:
        {{
            "response_code": 0,
            "results": [
                {{
                    "category": "Topic category",
                    "type": "multiple",
                    "difficulty": "easy/medium/hard",
                    "question": "The question text",
                    "correct_answer": "The correct answer",
                    "incorrect_answers": ["Wrong answer 1", "Wrong answer 2", "Wrong answer 3"]
                }},
                ... more questions
            ]
        }}
        
        Ensure the response is valid JSON that can be parsed directly.
        """

        # Call Gemini API without specifying response_schema
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        
        # Extract the JSON from the response text
        # Looking for JSON content between possible explanation text
        response_text = response.text
        
        # Try to find JSON in the response
        try:
            # First attempt: Try parsing the entire response as JSON
            quiz_data = json.loads(response_text)
        except json.JSONDecodeError:
            # Second attempt: Try to find JSON between triple backticks
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                try:
                    quiz_data = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    raise ValueError("Could not parse JSON from response")
            else:
                # Third attempt: Try to find any JSON-like structure
                json_match = re.search(r'(\{[\s\S]*\})', response_text)
                if json_match:
                    try:
                        quiz_data = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        raise ValueError("Could not parse JSON from response")
                else:
                    raise ValueError("No JSON structure found in response")
        
        # Validate the structure of quiz_data
        if "results" not in quiz_data:
            raise ValueError("Response does not contain 'results' field")
        
        return {
            "success": True,
            "questions": quiz_data["results"]
        }
        
    except Exception as e:
        print(f"Error generating quiz: {str(e)}")
        return {
            "success": False,
            "message": f"Failed to generate quiz: {str(e)}",
            "questions": []
        }

@app.post("/api/verify-quiz", response_model=QuizVerificationResponse)
async def verify_quiz(request: QuizAnswerRequest):
    try:
        # Extract user answers
        user_answers = request.answers
        total_questions = len(user_answers)
        
        if total_questions == 0:
            return {
                "success": False,
                "score": 0,
                "totalQuestions": 0,
                "passed": False,
                "message": "No quiz answers provided"
            }
        
        # Count correct answers
        correct_answers = sum(1 for answer in user_answers if answer.get("is_correct", False))
        
        # Calculate score percentage
        score_percentage = correct_answers / total_questions
        
        # Determine if quiz is passed (60% threshold)
        passed = score_percentage >= 0.6
        
        # If quiz is passed, calculate and award tokens
        reward_amount = 0
        tx_hash = None
        
        if passed:
            # Calculate reward based on difficulty and number of questions
            base_reward = 5  # Base reward for completing a quiz
            difficulty_multiplier = 1.0  # Default multiplier
            
            # Adjust multiplier based on average difficulty
            difficulties = [answer.get("difficulty", "medium") for answer in user_answers]
            difficulty_values = {"easy": 0.8, "medium": 1.0, "hard": 1.5}
            avg_difficulty = sum(difficulty_values.get(d, 1.0) for d in difficulties) / len(difficulties)
            
            # Calculate final reward
            reward_amount = int(base_reward * avg_difficulty * (score_percentage ** 0.5))
            
            # Transfer tokens to user
            transfer_result = await transfer_tokens(request.walletAddress, reward_amount)
            
            if not transfer_result["success"]:
                return {
                    "success": False,
                    "score": correct_answers,
                    "totalQuestions": total_questions,
                    "passed": passed,
                    "message": f"Failed to transfer tokens: {transfer_result.get('error', 'Unknown error')}"
                }
            
            tx_hash = transfer_result["txHash"]
            
            # Update user XP in Firebase if available
            if db:
                try:
                    xp_reward = reward_amount * 10  # 10 XP per token as a simple formula
                    user_ref = db.collection("users").document(request.walletAddress.lower())
                    user_doc = user_ref.get()
                    
                    if user_doc.exists:
                        user_data = user_doc.to_dict()
                        current_xp = user_data.get("xp", 0)
                        new_xp = current_xp + xp_reward
                        
                        # Update user XP
                        user_ref.update({
                            "xp": new_xp,
                            "tokensEarned": user_data.get("tokensEarned", 0) + reward_amount
                        })
                except Exception as db_error:
                    print(f"Firebase update error (non-critical): {db_error}")
        
        return {
            "success": True,
            "score": correct_answers,
            "totalQuestions": total_questions,
            "passed": passed,
            "reward": reward_amount if passed else None,
            "txHash": tx_hash,
            "message": "Quiz verification completed successfully"
        }
            
    except Exception as e:
        print(f"Error verifying quiz: {e}")
        return {
            "success": False,
            "score": 0,
            "totalQuestions": 0,
            "passed": False,
            "message": f"An error occurred during verification: {str(e)}"
        }

#############################################
# LeetCode Verification Endpoints
#############################################

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "LeetCode and Quiz verification service is running"}

# Endpoint to verify LeetCode task completion
@app.post("/api/verify/leetcode-task")
async def verify_leetcode_task(request: VerifyLeetCodeTaskRequest):
    try:
        # Extract problem slug from URL
        problem_slug = extract_problem_slug(request.problemUrl)
        if not problem_slug:
            return {
                "success": False,
                "message": "Invalid LeetCode problem URL"
            }
        
        # Get user's recent accepted submissions
        async with httpx.AsyncClient() as client:
            response = await client.post(
                LEETCODE_API,
                json={
                    "query": RECENT_AC_SUBMISSIONS_QUERY,
                    "variables": {
                        "username": request.leetcodeUsername,
                        "limit": 20  # Get last 20 accepted submissions
                    }
                },
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
                timeout=10.0
            )
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "message": f"Failed to fetch LeetCode submissions: {response.text}"
                }
            
            data = response.json()
            submissions = data.get("data", {}).get("recentAcSubmissionList", [])
            
            # Check if the problem has been solved
            solved_submission = None
            for submission in submissions:
                if submission["titleSlug"] == problem_slug:
                    solved_submission = submission
                    break
            
            if not solved_submission:
                return {
                    "success": False,
                    "message": "Problem not solved yet"
                }
            
            # If task creation time is provided, check if the solution was submitted after the task was created
            if request.taskCreatedAt:
                task_created_timestamp = int(datetime.fromisoformat(request.taskCreatedAt.replace('Z', '+00:00')).timestamp())
                submission_timestamp = int(solved_submission["timestamp"])
                
                if submission_timestamp < task_created_timestamp:
                    return {
                        "success": False,
                        "message": "Problem was solved before the task was created"
                    }
            
            # Get problem difficulty to calculate reward
            difficulty_response = await client.post(
                LEETCODE_API,
                json={
                    "query": PROBLEM_DIFFICULTY_QUERY,
                    "variables": {"titleSlug": problem_slug}
                },
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
                timeout=10.0
            )
            
            difficulty_data = difficulty_response.json()
            problem_data = difficulty_data.get("data", {}).get("question", {})
            difficulty = problem_data.get("difficulty", "Easy")
            
            # Calculate reward based on difficulty
            reward_amount = calculate_reward(difficulty)
            
            # Transfer tokens to user
            transfer_result = await transfer_tokens(request.walletAddress, reward_amount)
            
            if not transfer_result["success"]:
                return {
                    "success": False,
                    "message": f"Failed to transfer tokens: {transfer_result.get('error', 'Unknown error')}"
                }
            
            # Update user XP in Firebase if available
            if db:
                try:
                    xp_reward = reward_amount * 10  # 10 XP per token as a simple formula
                    user_ref = db.collection("users").document(request.walletAddress.lower())
                    user_doc = user_ref.get()
                    
                    if user_doc.exists:
                        user_data = user_doc.to_dict()
                        current_xp = user_data.get("xp", 0)
                        new_xp = current_xp + xp_reward
                        
                        # Update user XP
                        user_ref.update({
                            "xp": new_xp,
                            "tokensEarned": user_data.get("tokensEarned", 0) + reward_amount
                        })
                except Exception as db_error:
                    print(f"Firebase update error (non-critical): {db_error}")
            
            return {
                "success": True,
                "message": f"Congratulations! You've been rewarded {reward_amount} $TASK tokens for solving '{solved_submission['title']}'",
                "reward": reward_amount,
                "problem": {
                    "title": solved_submission["title"],
                    "difficulty": difficulty
                },
                "txHash": transfer_result["txHash"]
            }
            
    except Exception as e:
        print(f"Error verifying LeetCode task: {e}")
        return {
            "success": False,
            "message": f"An error occurred during verification: {str(e)}"
        }

# LeetCode account verification endpoint
@app.post("/api/verify/leetcode")
async def verify_leetcode(request: LeetCodeVerificationRequest, authorization: Optional[str] = None):
    try:
        username = request.username
        
        if not username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="LeetCode username is required"
            )
        
        # Extract wallet address from authorization header
        wallet_address = None
        if authorization and authorization.startswith("Bearer "):
            wallet_address = authorization.split("Bearer ")[1].lower()
        
        if not wallet_address:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: No valid wallet address provided"
            )
        
        print(f"Attempting to verify LeetCode username: {username} for wallet: {wallet_address}")
        
        # Get user data from Firebase to retrieve the verification token
        if db:
            try:
                user_ref = db.collection("users").document(wallet_address)
                user_doc = user_ref.get()
                
                if not user_doc.exists:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="User not found"
                    )
                
                user_data = user_doc.to_dict()
                verification_token = user_data.get("verificationToken")
                
                if not verification_token:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Verification token not found for user"
                    )
                
                # Call LeetCode GraphQL API with proper headers
                async with httpx.AsyncClient() as client:
                    leetcode_response = await client.post(
                        "https://leetcode.com/graphql",
                        json={
                            "query": USER_PROFILE_QUERY,
                            "variables": {"username": username}
                        },
                        headers={
                            "Content-Type": "application/json",
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                        },
                        timeout=10.0
                    )
                
                response_data = leetcode_response.json()
                matched_user = response_data.get("data", {}).get("matchedUser")
                
                if not matched_user:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="LeetCode user not found"
                    )
                
                # Check if the verification token is in the user's bio
                about_me = matched_user.get("profile", {}).get("aboutMe", "")
                is_verified = verification_token in about_me
                
                if not is_verified:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Verification failed. Please ensure the token is in your LeetCode bio."
                    )
                
                # Update the user's verification status in Firebase
                user_ref.update({
                    "leetcodeVerified": True,
                    "leetcodeUsername": username
                })
                
                # Get the updated user data
                updated_user_doc = user_ref.get()
                updated_user = updated_user_doc.to_dict()
                
                return {
                    "success": True,
                    "message": "LeetCode account verified successfully",
                    "user": updated_user
                }
            except HTTPException as http_ex:
                raise http_ex
            except Exception as db_error:
                print(f"Database error: {db_error}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Database error: {str(db_error)}"
                )
        else:
            # Fallback if Firebase is not available
            return {
                "success": True,
                "message": "LeetCode account verified successfully (Firebase unavailable)",
                "user": {
                    "leetcodeVerified": True,
                    "leetcodeUsername": username
                }
            }
            
    except HTTPException as http_ex:
        raise http_ex
    except Exception as error:
        print(f"Verification error: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during verification: {str(error)}"
        )

# Alternative endpoint that accepts wallet address in the request body
@app.post("/api/verify/leetcode/alt")
async def verify_leetcode_alt(request: AlternativeVerificationRequest):
    try:
        username = request.username
        wallet_address = request.walletAddress
        
        if not username or not wallet_address:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="LeetCode username and wallet address are required"
            )
        
        # Normalize wallet address
        normalized_address = wallet_address.lower()
        print(f"Alt endpoint: Attempting to verify LeetCode username: {username} for wallet: {normalized_address}")
        
        # Get user data from Firebase to retrieve the verification token
        if db:
            try:
                user_ref = db.collection("users").document(normalized_address)
                user_doc = user_ref.get()
                
                if not user_doc.exists:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="User not found"
                    )
                
                user_data = user_doc.to_dict()
                verification_token = user_data.get("verificationToken")
                
                if not verification_token:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Verification token not found for user"
                    )
                
                # Call LeetCode GraphQL API with proper headers
                async with httpx.AsyncClient() as client:
                    leetcode_response = await client.post(
                        "https://leetcode.com/graphql",
                        json={
                            "query": USER_PROFILE_QUERY,
                            "variables": {"username": username}
                        },
                        headers={
                            "Content-Type": "application/json",
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                        },
                        timeout=10.0
                    )
                
                response_data = leetcode_response.json()
                matched_user = response_data.get("data", {}).get("matchedUser")
                
                if not matched_user:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="LeetCode user not found"
                    )
                
                # Check if the verification token is in the user's bio
                about_me = matched_user.get("profile", {}).get("aboutMe", "")
                is_verified = verification_token in about_me
                
                if not is_verified:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Verification failed. Please ensure the token is in your LeetCode bio."
                    )
                
                # Update the user's verification status in Firebase
                user_ref.update({
                    "leetcodeVerified": True,
                    "leetcodeUsername": username
                })
                
                # Get the updated user data
                updated_user_doc = user_ref.get()
                updated_user = updated_user_doc.to_dict()
                
                return {
                    "success": True,
                    "message": "LeetCode account verified successfully",
                    "user": updated_user
                }
            except HTTPException as http_ex:
                raise http_ex
            except Exception as db_error:
                print(f"Database error: {db_error}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Database error: {str(db_error)}"
                )
        else:
            # Fallback if Firebase is not available
            return {
                "success": True,
                "message": "LeetCode account verified successfully (Firebase unavailable)",
                "user": {
                    "leetcodeVerified": True,
                    "leetcodeUsername": username
                }
            }
    except HTTPException as http_ex:
        raise http_ex
    except Exception as error:
        print(f"Verification error: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during verification: {str(error)}"
        )

# Start the server with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)