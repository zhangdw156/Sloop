"""
API结构化组织模块
支持树形和图形API组织方式，便于智能采样
"""

from typing import Dict, List, Any, Optional, Set
from abc import ABC, abstractmethod
import networkx as nx
from collections import defaultdict
import json


class APIStructure(ABC):
    """API结构抽象基类"""

    @abstractmethod
    def get_all_apis(self) -> List[Dict[str, Any]]:
        """获取所有API"""
        pass

    @abstractmethod
    def get_related_apis(self, api_name: str, max_related: int = 5) -> List[Dict[str, Any]]:
        """获取相关API"""
        pass

    @abstractmethod
    def sample_apis(self, k: int, strategy: str = "random") -> List[Dict[str, Any]]:
        """采样API"""
        pass


class TreeAPIStructure(APIStructure):
    """
    树形API结构
    按功能类别组织API，形成树形结构
    """

    def __init__(self, apis: List[Dict[str, Any]]):
        self.apis = apis
        self.api_map = {api['name']: api for api in apis}
        self.category_tree = self._build_category_tree()

    def _build_category_tree(self) -> Dict[str, List[Dict[str, Any]]]:
        """构建类别树"""
        tree = defaultdict(list)
        for api in self.apis:
            # 从API描述中提取类别，或者使用默认类别
            category = self._extract_category(api)
            tree[category].append(api)
        return dict(tree)

    def _extract_category(self, api: Dict[str, Any]) -> str:
        """从API信息中提取类别"""
        # 优先使用显式category字段
        if 'category' in api:
            return api['category']

        # 从description中提取关键词
        description = api.get('description', '').lower()

        # 预定义的类别映射
        category_keywords = {
            'weather': ['weather', 'temperature', 'forecast', 'climate'],
            'search': ['search', 'find', 'query', 'lookup'],
            'communication': ['email', 'message', 'chat', 'notification'],
            'data': ['database', 'storage', 'file', 'document'],
            'finance': ['payment', 'transaction', 'account', 'money'],
            'travel': ['booking', 'reservation', 'travel', 'hotel', 'flight'],
            'social': ['user', 'profile', 'social', 'friend'],
        }

        for category, keywords in category_keywords.items():
            if any(keyword in description for keyword in keywords):
                return category

        return 'general'

    def get_all_apis(self) -> List[Dict[str, Any]]:
        return self.apis

    def get_related_apis(self, api_name: str, max_related: int = 5) -> List[Dict[str, Any]]:
        """获取同类别API"""
        if api_name not in self.api_map:
            return []

        target_category = self._extract_category(self.api_map[api_name])
        related = [
            api for api in self.category_tree.get(target_category, [])
            if api['name'] != api_name
        ]
        return related[:max_related]

    def sample_apis(self, k: int, strategy: str = "random") -> List[Dict[str, Any]]:
        """采样API"""
        import random

        if strategy == "balanced":
            # 均衡采样：从不同类别中各选一些
            sampled = []
            categories = list(self.category_tree.keys())
            apis_per_category = max(1, k // len(categories))

            for category in categories:
                category_apis = self.category_tree[category]
                sampled.extend(random.sample(
                    category_apis,
                    min(apis_per_category, len(category_apis))
                ))

            # 如果还不够，随机补充
            if len(sampled) < k:
                remaining = [api for api in self.apis if api not in sampled]
                sampled.extend(random.sample(
                    remaining,
                    min(k - len(sampled), len(remaining))
                ))

            return sampled[:k]

        elif strategy == "tree_walk":
            # 树游走采样：从一个类别开始，游走到相关类别
            return self._sample_tree_walk(k)

        else:  # random
            return random.sample(self.apis, min(k, len(self.apis)))

    def _sample_tree_walk(self, k: int) -> List[Dict[str, Any]]:
        """树游走采样策略"""
        import random

        sampled = []
        visited_categories = set()

        # 选择一个起始类别
        start_category = random.choice(list(self.category_tree.keys()))
        current_category = start_category
        visited_categories.add(current_category)

        # 从当前类别采样
        current_apis = self.category_tree[current_category]
        sample_size = min(random.randint(1, 3), len(current_apis))
        sampled.extend(random.sample(current_apis, sample_size))

        # 游走到相关类别
        while len(sampled) < k and len(visited_categories) < len(self.category_tree):
            # 找到相似的类别（基于关键词相似性）
            similar_categories = self._find_similar_categories(current_category, visited_categories)

            if not similar_categories:
                break

            # 选择下一个类别
            next_category = random.choice(similar_categories)
            visited_categories.add(next_category)
            current_category = next_category

            # 从新类别采样
            next_apis = [api for api in self.category_tree[next_category] if api not in sampled]
            if next_apis:
                sample_size = min(random.randint(1, 2), len(next_apis))
                sampled.extend(random.sample(next_apis, sample_size))

        # 如果还不够，随机补充
        if len(sampled) < k:
            remaining = [api for api in self.apis if api not in sampled]
            sampled.extend(random.sample(
                remaining,
                min(k - len(sampled), len(remaining))
            ))

        return sampled[:k]

    def _find_similar_categories(self, category: str, visited: set) -> List[str]:
        """找到相似的类别"""
        # 预定义的类别相似性
        category_similarities = {
            'weather': ['travel'],
            'travel': ['weather', 'search'],
            'search': ['travel', 'data'],
            'communication': ['business'],
            'finance': ['business', 'data'],
            'data': ['search', 'finance'],
            'social': ['communication']
        }

        similar = category_similarities.get(category, [])
        # 排除已访问的类别
        return [cat for cat in similar if cat not in visited and cat in self.category_tree]

    def get_categories(self) -> List[str]:
        """获取所有类别"""
        return list(self.category_tree.keys())


class GraphAPIStructure(APIStructure):
    """
    图形API结构
    使用图论表示API间的依赖关系
    """

    def __init__(self, apis: List[Dict[str, Any]], relationships: Optional[List[Dict[str, Any]]] = None):
        self.apis = apis
        self.api_map = {api['name']: api for api in apis}
        self.graph = self._build_graph(relationships or [])

    def _build_graph(self, relationships: List[Dict[str, Any]]) -> nx.Graph:
        """构建API关系图"""
        G = nx.Graph()

        # 添加节点
        for api in self.apis:
            G.add_node(api['name'], **api)

        # 添加边（关系）
        for rel in relationships:
            api_from = rel.get('from')
            api_to = rel.get('to')
            rel_type = rel.get('type', 'related')

            if api_from in self.api_map and api_to in self.api_map:
                # 移除type键以避免冲突，然后添加边
                edge_attrs = {k: v for k, v in rel.items() if k not in ['from', 'to']}
                edge_attrs['type'] = rel_type
                G.add_edge(api_from, api_to, **edge_attrs)

        # 如果没有显式关系，基于描述自动构建关系
        if not relationships:
            self._build_auto_relationships(G)

        return G

    def _build_auto_relationships(self, graph: nx.Graph):
        """自动构建API间关系"""
        # 基于共同关键词建立关系
        for i, api1 in enumerate(self.apis):
            for api2 in self.apis[i+1:]:
                if self._are_related(api1, api2):
                    graph.add_edge(api1['name'], api2['name'], type='auto_related')

    def _are_related(self, api1: Dict[str, Any], api2: Dict[str, Any]) -> bool:
        """判断两个API是否相关"""
        desc1 = api1.get('description', '').lower()
        desc2 = api2.get('description', '').lower()

        # 共同关键词
        common_words = set(desc1.split()) & set(desc2.split())
        return len(common_words) >= 2

    def get_all_apis(self) -> List[Dict[str, Any]]:
        return self.apis

    def get_related_apis(self, api_name: str, max_related: int = 5) -> List[Dict[str, Any]]:
        """基于图关系获取相关API"""
        if api_name not in self.graph:
            return []

        # 获取直接邻居
        neighbors = list(self.graph.neighbors(api_name))
        related_apis = [self.api_map[name] for name in neighbors if name in self.api_map]

        return related_apis[:max_related]

    def sample_apis(self, k: int, strategy: str = "random") -> List[Dict[str, Any]]:
        """采样API"""
        import random

        if strategy == "connected":
            # 连通采样：选择一个种子API，然后扩展到其邻居
            if not self.apis:
                return []

            # 选择度最大的节点作为种子
            seed_node = max(self.graph.degree(), key=lambda x: x[1])[0]
            sampled_names = {seed_node}

            # 扩展到邻居
            neighbors = list(self.graph.neighbors(seed_node))
            sampled_names.update(neighbors[:k-1])

            return [self.api_map[name] for name in sampled_names if name in self.api_map]

        else:  # random
            return random.sample(self.apis, min(k, len(self.apis)))


class APICollection:
    """
    API集合工厂类
    自动选择合适的结构化方式
    """

    def __init__(self,
                 apis: List[Dict[str, Any]],
                 structure_type: str = "auto",
                 relationships: Optional[List[Dict[str, Any]]] = None):
        """
        初始化API集合

        Args:
            apis: API定义列表
            structure_type: 结构类型 ("tree", "graph", "auto")
            relationships: API关系定义（用于graph类型）
        """
        self.apis = apis

        if structure_type == "graph" or (structure_type == "auto" and relationships):
            self.structure = GraphAPIStructure(apis, relationships)
        else:
            self.structure = TreeAPIStructure(apis)

    def get_all_apis(self) -> List[Dict[str, Any]]:
        return self.structure.get_all_apis()

    def get_related_apis(self, api_name: str, max_related: int = 5) -> List[Dict[str, Any]]:
        return self.structure.get_related_apis(api_name, max_related)

    def sample_apis(self, k: int, strategy: str = "random") -> List[Dict[str, Any]]:
        return self.structure.sample_apis(k, strategy)

    def get_structure_info(self) -> Dict[str, Any]:
        """获取结构信息"""
        if isinstance(self.structure, TreeAPIStructure):
            return {
                "type": "tree",
                "categories": self.structure.get_categories(),
                "total_apis": len(self.apis)
            }
        else:
            return {
                "type": "graph",
                "nodes": len(self.structure.graph.nodes()),
                "edges": len(self.structure.graph.edges()),
                "total_apis": len(self.apis)
            }


def load_apis_from_file(file_path: str) -> List[Dict[str, Any]]:
    """从文件加载API定义"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
