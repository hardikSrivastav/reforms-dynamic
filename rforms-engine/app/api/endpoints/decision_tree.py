from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, List, Optional
from app.db.mongodb import get_decision_trees_collection
from app.db.redis import delete_cache
from app.models.question import DecisionTree
from loguru import logger
import uuid
from datetime import datetime

router = APIRouter()

@router.get("/decision-trees", response_model=List[Dict[str, Any]])
async def list_decision_trees():
    """List all decision trees."""
    collection = get_decision_trees_collection()
    trees = []
    
    try:
        # Get all decision trees (with limited fields for listing)
        cursor = collection.find(
            {},
            projection={"metric_id": 1, "metadata": 1, "created_at": 1, "updated_at": 1}
        )
        
        async for tree in cursor:
            # Remove MongoDB _id for serialization
            if "_id" in tree:
                del tree["_id"]
                
            trees.append(tree)
        
        return trees
    except Exception as e:
        logger.error(f"Error listing decision trees: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/decision-trees/{metric_id}", response_model=Dict[str, Any])
async def get_decision_tree(metric_id: str):
    """Get a decision tree for a specific metric."""
    collection = get_decision_trees_collection()
    
    try:
        tree = await collection.find_one({"metric_id": metric_id})
        
        if not tree:
            raise HTTPException(status_code=404, detail="Decision tree not found")
        
        # Remove MongoDB _id for serialization
        if "_id" in tree:
            del tree["_id"]
            
        return tree
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting decision tree: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/decision-trees", response_model=Dict[str, Any])
async def create_decision_tree(tree: DecisionTree):
    """Create or update a decision tree."""
    collection = get_decision_trees_collection()
    
    try:
        # Convert to dict for MongoDB
        tree_dict = tree.model_dump()
        
        # Check if a tree already exists for this metric
        existing_tree = await collection.find_one({"metric_id": tree.metric_id})
        
        if existing_tree:
            # Update existing tree
            tree_dict["updated_at"] = datetime.utcnow().isoformat()
            result = await collection.replace_one(
                {"metric_id": tree.metric_id},
                tree_dict
            )
            
            if result.modified_count == 0:
                raise HTTPException(status_code=500, detail="Failed to update decision tree")
                
            # Invalidate cache
            await delete_cache(f"decision_tree:{tree.metric_id}")
            
            return {"id": tree.id, "metric_id": tree.metric_id, "updated": True}
        else:
            # Create new tree
            tree_dict["created_at"] = datetime.utcnow().isoformat()
            tree_dict["updated_at"] = tree_dict["created_at"]
            
            result = await collection.insert_one(tree_dict)
            
            if not result.inserted_id:
                raise HTTPException(status_code=500, detail="Failed to create decision tree")
                
            return {"id": tree.id, "metric_id": tree.metric_id, "created": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating decision tree: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/decision-trees/{metric_id}")
async def delete_decision_tree(metric_id: str):
    """Delete a decision tree."""
    collection = get_decision_trees_collection()
    
    try:
        result = await collection.delete_one({"metric_id": metric_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Decision tree not found")
            
        # Invalidate cache
        await delete_cache(f"decision_tree:{metric_id}")
        
        return {"metric_id": metric_id, "deleted": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting decision tree: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/decision-trees/{metric_id}/nodes/{node_id}", response_model=Dict[str, Any])
async def update_tree_node(
    metric_id: str,
    node_id: str,
    node_data: Dict[str, Any] = Body(...)
):
    """Update a specific node in a decision tree."""
    collection = get_decision_trees_collection()
    
    try:
        # Check if the tree exists
        tree = await collection.find_one({"metric_id": metric_id})
        
        if not tree:
            raise HTTPException(status_code=404, detail="Decision tree not found")
        
        # Update the specific node
        result = await collection.update_one(
            {"metric_id": metric_id},
            {
                "$set": {
                    f"nodes.{node_id}": node_data,
                    "updated_at": datetime.utcnow().isoformat()
                }
            }
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=500, detail="Failed to update node")
            
        # Invalidate cache
        await delete_cache(f"decision_tree:{metric_id}")
        
        return {"metric_id": metric_id, "node_id": node_id, "updated": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating tree node: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 