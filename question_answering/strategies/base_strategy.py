import abc
import six
from typing import List, Any, Optional, Dict, Callable
from question_answering.qa_context import QAContext

@six.add_metaclass(abc.ABCMeta)
class BaseStrategy():
    
    @classmethod
    def strategy_name(cls):
        return 'base-strategy'

    def add_results(self, context: QAContext, results: List[Any]) -> None:
        context.add_results(self.strategy_name(), results)
    
    @classmethod
    def get_results(self, 
                    context: QAContext, 
                    strategies: List[type] = [], 
                    selector : Optional[str] = None,
                    selector_func : Callable[[Dict], str] = None) -> List[Any]:
        
        all_results = []
        
        keys = []
        
        if not strategies:
            keys.append(self.strategy_name())
        else:
            if isinstance(strategies, list):
                keys.extend(strategy.strategy_name() for strategy in strategies)
            else:
                keys.append(strategies.strategy_name())

        
        for key in keys:
            
            results = [
                result
                for results in context.results if results['key'] == key
                for result in results['results']    
            ]
            
            def get_by_alternative(result, alternatives):
                for alternative in alternatives:
                    if alternative in result:
                        return result[alternative]
                return None
            
            if selector:
                selector_parts = selector.split('/')
                for selector_part in selector_parts:
                    selector_part_alternatives = selector_part.split('|')
                    results = [get_by_alternative(result, selector_part_alternatives) for result in results]
                    
            if selector_func:
                results = [selector_func(result) for result in results]
                
            all_results.extend(results)
                
        return all_results
    
    @abc.abstractmethod
    def accept(self, context: QAContext) -> QAContext:
        return context