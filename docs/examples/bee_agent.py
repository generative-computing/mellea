"""
Example use case for BeeAI integration: utilizing a Mellea program to write an email with an IVF loop.
Also demo of RangeType to demonstrate random selection of a integer from a given range
"""
import os
import asyncio
import sys
from typing import Annotated

from mellea import MelleaSession, start_session
from mellea.stdlib.base import ChatContext, ModelOutputThunk

from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.stdlib.sampling.types import SamplingResult
from mellea.stdlib.sampling.base import Context
from mellea.stdlib.requirement import req, Requirement, simple_validate
#from cli.serve.bee_playform.types import RangeType
from bee_platform.bee_platform import bee_app


@bee_app
def mellea_func(m: MelleaSession, sender: str, recipient, subject: str, topic: str, sampling_iters : int = 3) -> tuple[ModelOutputThunk, Context] | SamplingResult:
    """
    Example email writing module that utilizes an IVR loop in Mellea to generate an email with a specific list of requirements.
    Inputs:
        sender: str
        recipient: str
        subject: str
	topic: str
    Output:
	sampling: tuple[ModelOutputThunk, Context] | SamplingResult
    """
    requirements = [
        req("Be formal."),
        req("Be funny."),
	req(f"Make sure that the email is from {sender}, is towards {recipient}, has {subject} as the subject, and is focused on {topic} as a topic"),
        Requirement("Use less than 100 words.", 
                   validation_fn=simple_validate(lambda o: len(o.split()) < 100))
    ]
    sampling = m.instruct(f"Write an email from {sender}. Subject of email is {subject}. Name of recipient is {recipient}. Topic of email should be {topic}.", requirements=requirements, strategy=RejectionSamplingStrategy(loop_budget=1), return_sampling_results=True)
    
    return sampling





