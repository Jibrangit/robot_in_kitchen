import py_trees
import numpy as np
import matplotlib.pyplot as plt
from libraries.mapping import Mapper, MappingParams
from libraries.motion_planning import astar


class GeneratePath(py_trees.behaviour.Behaviour):
    def __init__(
        self,
        goal_position: tuple,
        display: bool,
        name: str = "GeneratePath",
    ):
        super(GeneratePath, self).__init__(name)
        self.logger.info("%s.__init__()" % (self.__class__.__name__))

        self._goal_position = goal_position
        self._mapper = Mapper(MappingParams("config/mapping_params.yaml"))

        self._blackboard = self.attach_blackboard_client()
        self._blackboard.register_key(
            key="current_plan", access=py_trees.common.Access.WRITE
        )
        self._blackboard.register_key(
            key="se2_pose", access=py_trees.common.Access.READ
        )

        self._plan = None
        self._display = display

    def _display_plan(self):
        if self._plan:
            plt.imshow(self._cspace)
            for position in self._plan:
                plt.plot(position[1], position[0], "r*")
                plt.pause(0.000001)

        plt.show()

    def initialise(self) -> None:
        self.logger.info(f"Planning path for behavior {self.name}")

        self.logger.info("%s.initialise()" % (self.name))
        try:
            self._cspace = np.load("maps/kitchen_cspace.npy")

        except OSError:
            self.logger.error("No Cspace available for path planning!")

        xw, yw, theta = self._blackboard.se2_pose

        self._home_position = (xw, yw)

        p_start = self._mapper.world2map(self._home_position[0], self._home_position[1])
        p_goal = self._mapper.world2map(self._goal_position[0], self._goal_position[1])

        self._plan = astar(self._cspace, p_start, p_goal)
        for idx, pose in enumerate(self._plan):
            self._plan[idx] = self._mapper.map2world(pose[0], pose[1])

        if self._display:
            self._display_plan()

    def update(self) -> py_trees.common.Status:

        if self._plan:
            self._blackboard.set(name="current_plan", value=self._plan, overwrite=True)

            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE
