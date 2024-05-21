# ruff: noqa
from functools import partial
from inspect import Parameter, Signature, signature
from typing import Any, Callable, Optional

from docstring_parser import parse
from pydantic import (
    BaseModel,
    ValidationError,
    create_model,
    model_serializer,
)
from pydantic.fields import FieldInfo

KEPT_PARAMS = ["env"]
EMPTY_MODEL = create_model("Empty")


class Action(BaseModel):
    """
    The core operational unit within the Crab system.

    This class stores parameters and return type definitions and can be easily converted
    into a JSON schema. It supports argument verification and includes a feature for
    retaining specific parameters.

    Attributes:
        name: The name of the action.
        entry: The actual entry function of the action.
        parameters: Definition of input parameters.
        returns: Definition of the return type.
        description: A clear and concise description of the function's purpose and
            behavior. Defaults to :code:`None`.
        kept_params: Parameters retained for internal use by the Crab system, such as
            :code:`env` for storing the current environment. These parameters do not
            appear in :attr:`parameters` and are automatically injected at runtime.
            Defaults to an empty dictionary.
        env_name: Specify the environment the action is associated with. Defaults to
            :code:`None`.
        local: Specify if the action should take in the local machine. Only use in
            remote mode. Defaults to :code:`False`.
    """

    name: str
    entry: Callable
    parameters: type[BaseModel]
    returns: type[BaseModel]
    description: Optional[str] = None
    kept_params: list[str] = []
    env_name: Optional[str] = None
    local: bool = False

    def __eq__(self, other):
        return super().__eq__(other)

    def __hash__(self):
        return hash(self.entry)

    def __call__(self, *args: Any, **kwargs: Any):
        """Sets default values for the action.

        Direct calling of the action will not actully call the function, yet set
        defaults values for the action, so the agent don't need to or only need to
        provide part of the parameters.

        This method has two mode, full setting and partial setting. Full setting mode is
        applied when the user provides positional arguments, where all the required
        parameters must be provide and the action parameters will be empty. While if
        only keyword arguments are provided, partial setting mode is applied, where the
        parameter model will not be changed but only change the default value of the
        parameters.

        Note:
            Full setting mode is not stable.
        """
        if args:
            # this is closed function
            result = self.model_copy(
                update={
                    "entry": partial(self.entry, *args, **kwargs),
                    "parameters": EMPTY_MODEL,
                }
            )
            if self.description is not None:
                result.description = self.description + f" Input: {args} {kwargs}"
            return result
        else:
            # or it should only contain kwargs
            for key in kwargs:
                # verify the kwargs exist
                if key not in self.parameters.model_fields:
                    raise ValueError(
                        f'"{key}" is not a parameter of action "{self.name}"'
                    )

            result = self.model_copy(
                update={
                    "entry": partial(self.entry, **kwargs),
                }
            )
            if self.description is not None:
                result.description = self.description + f" Input: {args} {kwargs}"
            return result

    def __rshift__(self, other_action: "Action") -> "Action":
        """Uses :obj:`>>` to pipe two actions together to form a new action.

        The returned action executes the actions from left to right. The output of the
        left action becomes the input to the right action, provided their parameters and
        return types are compatible.
        """
        required = other_action.get_required_params()
        if len(required) != 1:
            raise ValueError(
                "Return type of the former action must mathces the parameter type "
                "of the later action."
            )
        if set(self.kept_params) != set(other_action.kept_params):
            raise ValueError("Piped actions should have same kept parameters.")
        if self.env_name != other_action.env_name:
            raise ValueError("Piped actions should have same env_name.")
        if self.local != other_action.local:
            raise ValueError("Piped actions should have same `local` value.")

        a_entry = self.entry
        b_entry = other_action.entry
        kept_params = self.kept_params.copy()
        entry = lambda *args, **kwargs: b_entry(
            a_entry(*args, **kwargs),
            **{key: kwargs[key] for key in kwargs if key in kept_params},
        )
        return Action(
            name=f"{self.name}_pipe_{other_action.name}",
            description=f"First {self.description}. Then use the result of the "
            f"former as input, {other_action.description}",
            parameters=self.parameters,
            returns=other_action.returns,
            entry=entry,
            kept_params=self.kept_params,
            env_name=self.env_name,
            local=self.local,
        )

    def __add__(self, other_action: "Action") -> "Action":
        """Uses :obj:`+` to combine two actions sequetially to form a new action.

        The returned action executes the actions from left to right. Its return value
        will be the return value of the right action.

        Note:
            "+" operator only support two action with no required parameters.
        """
        self_required = self.get_required_params()
        other_required = other_action.get_required_params()
        if len(other_required) > 1 or len(self_required) > 1:
            raise ValueError(
                '"+" operator only support two action with no required parameters.'
            )
        if set(self.kept_params) != set(other_action.kept_params):
            raise ValueError("Combined actions should have same kept parameters.")
        if self.env_name != other_action.env_name:
            raise ValueError("Combined actions should have same env_name.")
        if self.local != other_action.local:
            raise ValueError("Piped actions should have same `local` value.")

        a_entry = self.entry
        b_entry = other_action.entry
        entry = lambda **kwargs: (a_entry(**kwargs), b_entry(**kwargs))[1]
        return Action(
            name=f"{self.name}_then_{other_action.name}",
            description=f"{self.description} Then, {other_action.description}",
            parameters=EMPTY_MODEL,
            returns=other_action.returns,
            entry=entry,
            kept_params=self.kept_params,
            env_name=self.env_name,
            local=self.local,
        )

    def run(self, **kwargs) -> Any:
        """Varifies the action parameters then runes the action."""
        if self.kept_params:
            raise RuntimeError("There are unassigned kept parameters.")
        try:
            kwargs = self.parameters(**kwargs).model_dump()
        except ValidationError:
            pass  # TODO: Exeception handle
        return self.entry(**kwargs)

    def set_kept_param(self, **params):
        kept_params = {key: params[key] for key in params if key in self.kept_params}
        result = self.model_copy()
        result.kept_params = []
        result.entry = partial(self.entry, **kept_params)
        return result

    def get_required_params(self) -> dict[str, FieldInfo]:
        return {
            name: info
            for name, info in self.parameters.model_fields.items()
            if info.is_required()
        }

    @model_serializer
    def to_openai_json_schema(self) -> dict:
        """Gets openai json schema from an action"""

        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters.model_json_schema(),
        }

    @classmethod
    def from_function(cls, func: Callable):
        """Generates an action from functions annotated by @action."""
        if func.__doc__ is None:
            # raise RuntimeError("The action must have a Google-style docstring.")
            parameters_descriptions = None
            func_description = None
            return_description = None
        else:
            docstring = parse(func.__doc__)
            parameters_descriptions = {
                param.arg_name: param.description for param in docstring.params
            }
            func_description = docstring.short_description or ""
            if docstring.long_description:
                func_description += "\n" + docstring.long_description
            if docstring.returns:
                return_description = docstring.returns.description
            else:
                return_description = None

        sign = signature(func)
        params = sign.parameters
        fields = {}
        kept_params = []
        for param_name, p in params.items():
            # Don't add kept parameters in prameters' model
            if param_name in KEPT_PARAMS:
                kept_params.append(param_name)
                continue
            # Variable parameters are not supported
            if p.kind in [Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD]:
                continue
            # If the parameter type is not specified, it defaults to typing.Any
            annotation = Any if p.annotation is Parameter.empty else p.annotation
            # Check if the parameter has a description
            param_description = None
            if parameters_descriptions is not None:
                param_description = parameters_descriptions.get(param_name, None)
            # Check if the parameter has a default value
            if p.default is Parameter.empty:
                fields[param_name] = (
                    annotation,
                    FieldInfo(description=param_description),
                )
            else:
                fields[param_name] = (annotation, FieldInfo(default=p.default))
        model: type[BaseModel] = create_model(func.__name__, **fields)  # type: ignore

        # insert return to parameters
        return_annotation = (
            Any if sign.return_annotation == Signature.empty else sign.return_annotation
        )
        return_model: type[BaseModel] = create_model(
            func.__name__ + "_return",
            returns=(
                return_annotation or None,
                FieldInfo(description=return_description, init=False),  # type: ignore
            ),
        )

        action = cls(
            name=func.__name__,
            entry=func,
            parameters=model,
            returns=return_model,
            description=func_description,
            kept_params=kept_params,
        )
        return action


def _decorator(func, cls: type[Action], options: Optional[dict] = None) -> Action:
    action = cls.from_function(func)
    if options is not None:
        for key in options:
            setattr(action, key, options[key])

    return action


def action(*args: Callable, env_name: Optional[str] = None, local=False):
    """Use @action to create an Action from a function"""
    if args and callable(args[0]):
        return _decorator(args[0], Action)

    return lambda func: _decorator(func, Action, {"env_name": env_name, "local": local})
